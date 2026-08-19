"""Tests for scan sequence preflight validation across STOMP, OaK, IA, and Learning Signals."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAAgent,
    IAConfig,
)
from alberta_framework.core.learning_signals import (
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
)
from alberta_framework.core.oak import OaKAgent, OaKConfig
from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec


def _make_stomp_agent() -> tuple[STOMPAgent, STOMPConfig]:
    config = STOMPConfig(
        observation_dim=4,
        n_primitive_actions=2,
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=0.5,
            ),
        ),
    )
    return STOMPAgent(config), config


def _make_oak_agent() -> tuple[OaKAgent, OaKConfig]:
    stomp_cfg = STOMPConfig(
        observation_dim=4,
        n_primitive_actions=2,
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=0.5,
            ),
        ),
    )
    config = OaKConfig(stomp=stomp_cfg)
    return OaKAgent(config), config


def _make_ia_agent() -> tuple[IAAgent, IAConfig]:
    cerebellum_cfg = ExoCerebellumConfig(n_demons=2, obs_dim=4)
    oak_cfg = OaKConfig(
        stomp=STOMPConfig(
            observation_dim=4,
            n_primitive_actions=2,
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=0.5,
                ),
            ),
        )
    )
    config = IAConfig(cerebellum=cerebellum_cfg, cortex=oak_cfg)
    return IAAgent(config), config


# ---------------------------------------------------------------------------
# STOMP scan validation tests
# ---------------------------------------------------------------------------


def test_stomp_scan_rejects_non_1d_or_empty_rewards() -> None:
    agent, _ = _make_stomp_agent()
    state = agent.init(jr.key(0))
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="env_rewards must have shape \\(num_steps,\\)"):
        agent.scan(state, jnp.zeros((0,), dtype=jnp.float32), jnp.zeros((0, 4), dtype=jnp.float32))

    with pytest.raises(ValueError, match="env_rewards must have shape \\(num_steps,\\)"):
        agent.scan(state, jnp.zeros((3, 1), dtype=jnp.float32), next_obs)


def test_stomp_scan_rejects_mismatched_next_observations() -> None:
    agent, _ = _make_stomp_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="next_observations must have shape \\(3, 4\\)"):
        agent.scan(state, rewards, jnp.zeros((2, 4), dtype=jnp.float32))

    with pytest.raises(ValueError, match="next_observations must have shape \\(3, 4\\)"):
        agent.scan(state, rewards, jnp.zeros((3, 5), dtype=jnp.float32))


def test_stomp_scan_rejects_mismatched_discounts() -> None:
    agent, _ = _make_stomp_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="discounts must have shape \\(3,\\)"):
        agent.scan(state, rewards, next_obs, discounts=jnp.zeros((2,), dtype=jnp.float32))


def test_stomp_scan_rejects_mismatched_decision_obs() -> None:
    agent, _ = _make_stomp_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="decision_observations must have shape \\(3, 4\\)"):
        agent.scan(
            state,
            rewards,
            next_obs,
            decision_observations=jnp.zeros((3, 2), dtype=jnp.float32),
        )


def test_stomp_scan_rejects_mismatched_execution_boundaries() -> None:
    agent, _ = _make_stomp_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="execution_boundaries must have shape \\(3,\\)"):
        agent.scan(
            state,
            rewards,
            next_obs,
            execution_boundaries=jnp.zeros((2,), dtype=jnp.bool_),
        )

    with pytest.raises(TypeError, match="execution_boundaries must have dtype bool"):
        agent.scan(
            state,
            rewards,
            next_obs,
            execution_boundaries=jnp.zeros((3,), dtype=jnp.float32),
        )


def test_stomp_scan_rejects_mismatched_extended_action_masks() -> None:
    agent, config = _make_stomp_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)
    n_actions = config.n_total_actions

    with pytest.raises(
        ValueError,
        match=f"extended_action_masks must have shape \\(3, {n_actions}\\)",
    ):
        agent.scan(
            state,
            rewards,
            next_obs,
            extended_action_masks=jnp.ones((2, n_actions), dtype=jnp.bool_),
        )

    with pytest.raises(TypeError, match="extended_action_masks must have dtype bool"):
        agent.scan(
            state,
            rewards,
            next_obs,
            extended_action_masks=jnp.ones((3, n_actions), dtype=jnp.float32),
        )


def test_stomp_scan_matches_loop() -> None:
    agent, _ = _make_stomp_agent()
    init_state = agent.init(jr.key(0))
    initial = agent.start(init_state, jnp.ones((4,), dtype=jnp.float32))
    rewards = jnp.array([0.5, 1.0, -0.5], dtype=jnp.float32)
    next_obs = jnp.array(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
        dtype=jnp.float32,
    )

    scan_res = agent.scan(initial, rewards, next_obs)

    curr_state = initial
    td_errs = []
    for r, o in zip(rewards, next_obs, strict=True):
        upd = agent.update(curr_state, r, o)
        curr_state = upd.state
        td_errs.append(upd.td_error)

    np.testing.assert_allclose(
        np.asarray(scan_res.td_errors),
        np.asarray(jnp.stack(td_errs)),
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# OaK scan validation tests
# ---------------------------------------------------------------------------


def test_oak_scan_rejects_non_1d_or_empty_rewards() -> None:
    agent, _ = _make_oak_agent()
    state = agent.init(jr.key(0))
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="env_rewards must have shape \\(num_steps,\\)"):
        agent.scan(state, jnp.zeros((0,), dtype=jnp.float32), jnp.zeros((0, 4), dtype=jnp.float32))

    with pytest.raises(ValueError, match="env_rewards must have shape \\(num_steps,\\)"):
        agent.scan(state, jnp.zeros((3, 1), dtype=jnp.float32), next_obs)


def test_oak_scan_rejects_mismatched_next_observations() -> None:
    agent, _ = _make_oak_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="next_observations must have shape \\(3, 4\\)"):
        agent.scan(state, rewards, jnp.zeros((2, 4), dtype=jnp.float32))

    with pytest.raises(ValueError, match="next_observations must have shape \\(3, 4\\)"):
        agent.scan(state, rewards, jnp.zeros((3, 5), dtype=jnp.float32))


def test_oak_scan_rejects_mismatched_discounts() -> None:
    agent, _ = _make_oak_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="discounts must have shape \\(3,\\)"):
        agent.scan(state, rewards, next_obs, discounts=jnp.zeros((2,), dtype=jnp.float32))


def test_oak_scan_rejects_mismatched_decision_obs() -> None:
    agent, _ = _make_oak_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="decision_observations must have shape \\(3, 4\\)"):
        agent.scan(
            state,
            rewards,
            next_obs,
            decision_observations=jnp.zeros((3, 2), dtype=jnp.float32),
        )


def test_oak_scan_rejects_mismatched_execution_boundaries() -> None:
    agent, _ = _make_oak_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="execution_boundaries must have shape \\(3,\\)"):
        agent.scan(
            state,
            rewards,
            next_obs,
            execution_boundaries=jnp.zeros((2,), dtype=jnp.bool_),
        )

    with pytest.raises(TypeError, match="execution_boundaries must have dtype bool"):
        agent.scan(
            state,
            rewards,
            next_obs,
            execution_boundaries=jnp.zeros((3,), dtype=jnp.float32),
        )


def test_oak_scan_rejects_mismatched_extended_action_masks() -> None:
    agent, config = _make_oak_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)
    n_actions = config.stomp.n_total_actions

    with pytest.raises(
        ValueError,
        match=f"extended_action_masks must have shape \\(3, {n_actions}\\)",
    ):
        agent.scan(
            state,
            rewards,
            next_obs,
            extended_action_masks=jnp.ones((2, n_actions), dtype=jnp.bool_),
        )

    with pytest.raises(TypeError, match="extended_action_masks must have dtype bool"):
        agent.scan(
            state,
            rewards,
            next_obs,
            extended_action_masks=jnp.ones((3, n_actions), dtype=jnp.float32),
        )


# ---------------------------------------------------------------------------
# IA scan validation tests
# ---------------------------------------------------------------------------


def test_ia_scan_rejects_non_1d_or_empty_rewards() -> None:
    agent, _ = _make_ia_agent()
    state = agent.init(jr.key(0))
    obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="partner_rewards must have shape \\(num_steps,\\)"):
        agent.scan(
            state,
            jnp.zeros((0, 4), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0, 4), dtype=jnp.float32),
        )

    with pytest.raises(ValueError, match="partner_rewards must have shape \\(num_steps,\\)"):
        agent.scan(state, obs, jnp.zeros((3, 1), dtype=jnp.float32), obs)


def test_ia_scan_rejects_mismatched_partner_obs() -> None:
    agent, _ = _make_ia_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="partner_obs must have shape \\(3, 4\\)"):
        agent.scan(state, jnp.zeros((2, 4), dtype=jnp.float32), rewards, next_obs)

    with pytest.raises(ValueError, match="partner_obs must have shape \\(3, 4\\)"):
        agent.scan(state, jnp.zeros((3, 5), dtype=jnp.float32), rewards, next_obs)


def test_ia_scan_rejects_mismatched_partner_next_obs() -> None:
    agent, _ = _make_ia_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="partner_next_obs must have shape \\(3, 4\\)"):
        agent.scan(state, obs, rewards, jnp.zeros((2, 4), dtype=jnp.float32))

    with pytest.raises(ValueError, match="partner_next_obs must have shape \\(3, 4\\)"):
        agent.scan(state, obs, rewards, jnp.zeros((3, 5), dtype=jnp.float32))


def test_ia_scan_rejects_mismatched_discounts() -> None:
    agent, _ = _make_ia_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="discounts must have shape \\(3,\\)"):
        agent.scan(state, obs, rewards, obs, discounts=jnp.zeros((2,), dtype=jnp.float32))


def test_ia_scan_rejects_mismatched_decision_obs() -> None:
    agent, _ = _make_ia_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(
        ValueError,
        match="partner_decision_obs must have shape \\(3, 4\\)",
    ):
        agent.scan(
            state,
            obs,
            rewards,
            obs,
            partner_decision_obs=jnp.zeros((3, 2), dtype=jnp.float32),
        )


def test_ia_scan_rejects_mismatched_execution_boundaries() -> None:
    agent, _ = _make_ia_agent()
    state = agent.init(jr.key(0))
    rewards = jnp.zeros((3,), dtype=jnp.float32)
    obs = jnp.zeros((3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError, match="execution_boundaries must have shape \\(3,\\)"):
        agent.scan(
            state,
            obs,
            rewards,
            obs,
            execution_boundaries=jnp.zeros((2,), dtype=jnp.bool_),
        )

    with pytest.raises(TypeError, match="execution_boundaries must have dtype bool"):
        agent.scan(
            state,
            obs,
            rewards,
            obs,
            execution_boundaries=jnp.zeros((3,), dtype=jnp.float32),
        )


# ---------------------------------------------------------------------------
# Learning signals scan validation tests
# ---------------------------------------------------------------------------


def test_learning_signals_scan_rejects_mismatched_shapes() -> None:
    config = LearningSignalEstimatorConfig(ensemble_size=3, target_dim=2)
    estimator = LearningSignalEstimator(config)
    state = estimator.init()

    means = jnp.zeros((4, 3, 2), dtype=jnp.float32)
    variances = jnp.zeros((4, 3, 2), dtype=jnp.float32)
    targets = jnp.zeros((4, 2), dtype=jnp.float32)
    losses = jnp.zeros((4,), dtype=jnp.float32)

    # Empty sequence
    with pytest.raises(ValueError, match="member_means sequence must be non-empty"):
        estimator.scan(
            state,
            jnp.zeros((0, 3, 2), dtype=jnp.float32),
            jnp.zeros((0, 3, 2), dtype=jnp.float32),
            jnp.zeros((0, 2), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
        )

    # Rank != 3
    with pytest.raises(ValueError, match="member_means sequence must have rank 3"):
        estimator.scan(state, jnp.zeros((4, 3), dtype=jnp.float32), variances, targets, losses)

    # Variances mismatch
    with pytest.raises(ValueError, match="predicted_aleatoric_variances sequence must have shape"):
        estimator.scan(
            state,
            means,
            jnp.zeros((3, 3, 2), dtype=jnp.float32),
            targets,
            losses,
        )

    # Targets mismatch
    with pytest.raises(ValueError, match="observed_targets must have shape"):
        estimator.scan(
            state,
            means,
            variances,
            jnp.zeros((4, 3), dtype=jnp.float32),
            losses,
        )

    # Losses mismatch
    with pytest.raises(ValueError, match="observed_losses must have shape"):
        estimator.scan(
            state,
            means,
            variances,
            targets,
            jnp.zeros((2,), dtype=jnp.float32),
        )


def test_learning_signals_scan_matches_observe_loop() -> None:
    config = LearningSignalEstimatorConfig(ensemble_size=3, target_dim=2)
    estimator = LearningSignalEstimator(config)
    state = estimator.init()

    key = jr.key(42)
    k1, k2, k3, k4 = jr.split(key, 4)
    means = jr.normal(k1, (5, 3, 2))
    variances = jnp.abs(jr.normal(k2, (5, 3, 2))) + 0.1
    targets = jr.normal(k3, (5, 2))
    losses = jnp.abs(jr.normal(k4, (5,)))

    final_state_scan, signals_scan = estimator.scan(
        state,
        means,
        variances,
        targets,
        losses,
    )

    curr_state = state
    signals_list = []
    for m, v, t, l_val in zip(means, variances, targets, losses, strict=True):
        curr_state, sig = estimator.observe(curr_state, m, v, t, l_val)
        signals_list.append(sig)

    np.testing.assert_allclose(
        np.asarray(final_state_scan.step_count),
        np.asarray(curr_state.step_count),
    )
    np.testing.assert_allclose(
        np.asarray(signals_scan.epistemic_surprise),
        np.asarray(jnp.stack([s.epistemic_surprise for s in signals_list])),
        rtol=1e-5,
        atol=1e-5,
    )
