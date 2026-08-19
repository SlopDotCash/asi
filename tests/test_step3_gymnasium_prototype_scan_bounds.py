# mypy: disable-error-code="call-arg,no-untyped-def,untyped-decorator"
"""Tests for scan sequence bounds and array dimension validation.

Covers Step 3, Gymnasium stream learning, and PrototypeAgent.
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.types import LearnerState
from alberta_framework.steps.step3 import (
    Step3HordeConfig,
    make_step3_horde,
    run_step3_scan,
)
from alberta_framework.streams.gymnasium import (
    learn_from_trajectory,
    learn_from_trajectory_normalized,
)

# =============================================================================
# Step 3 Horde scan bounds and dimensions
# =============================================================================


class TestStep3ScanValidation:
    @pytest.fixture
    def horde_and_state(self):
        cfg = Step3HordeConfig(
            gammas=(0.0, 0.9),
            lamdas=(0.0, 0.5),
            hidden_sizes=(16,),
        )
        horde = make_step3_horde(cfg)
        key = jr.key(42)
        state = horde.init(feature_dim=4, key=key)
        return horde, state

    def test_rejects_non_horde_type(self, horde_and_state) -> None:
        _, state = horde_and_state
        features = jnp.zeros((8, 4), dtype=jnp.float32)
        cumulants = jnp.zeros((8, 2), dtype=jnp.float32)
        with pytest.raises(TypeError, match="horde must be an exact HordeLearner"):
            run_step3_scan(
                cast(Any, object()),
                state,
                features,
                cumulants,
                features,
            )

    def test_rejects_non_state_type(self, horde_and_state) -> None:
        horde, _ = horde_and_state
        features = jnp.zeros((8, 4), dtype=jnp.float32)
        cumulants = jnp.zeros((8, 2), dtype=jnp.float32)
        with pytest.raises(TypeError, match="state must be an exact MultiHeadMLPState"):
            run_step3_scan(
                horde,
                cast(Any, object()),
                features,
                cumulants,
                features,
            )

    def test_rejects_zero_steps(self, horde_and_state) -> None:
        horde, state = horde_and_state
        features = jnp.zeros((0, 4), dtype=jnp.float32)
        cumulants = jnp.zeros((0, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="between 1 and signed-int32 steps"):
            run_step3_scan(
                horde,
                state,
                features,
                cumulants,
                features,
            )

    def test_rejects_zero_feature_dim(self, horde_and_state) -> None:
        horde, state = horde_and_state
        features = jnp.zeros((8, 0), dtype=jnp.float32)
        cumulants = jnp.zeros((8, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="at least one feature column"):
            run_step3_scan(
                horde,
                state,
                features,
                cumulants,
                features,
            )

    def test_rejects_mismatched_next_features(self, horde_and_state) -> None:
        horde, state = horde_and_state
        features = jnp.zeros((8, 4), dtype=jnp.float32)
        next_features = jnp.zeros((8, 5), dtype=jnp.float32)
        cumulants = jnp.zeros((8, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="next_features must match features"):
            run_step3_scan(
                horde,
                state,
                features,
                cumulants,
                next_features,
            )

    def test_rejects_mismatched_cumulants_steps(self, horde_and_state) -> None:
        horde, state = horde_and_state
        features = jnp.zeros((8, 4), dtype=jnp.float32)
        cumulants = jnp.zeros((6, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="cumulants must match steps and configured demons"):
            run_step3_scan(
                horde,
                state,
                features,
                cumulants,
                features,
            )

    def test_rejects_mismatched_cumulants_demons(self, horde_and_state) -> None:
        horde, state = horde_and_state
        features = jnp.zeros((8, 4), dtype=jnp.float32)
        cumulants = jnp.zeros((8, 3), dtype=jnp.float32)
        with pytest.raises(ValueError, match="cumulants must match steps and configured demons"):
            run_step3_scan(
                horde,
                state,
                features,
                cumulants,
                features,
            )

    def test_rejects_state_feature_dim_mismatch(self, horde_and_state) -> None:
        horde, state = horde_and_state  # state initialized for feature_dim=4
        features = jnp.zeros((8, 6), dtype=jnp.float32)
        cumulants = jnp.zeros((8, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="state feature dimension .* does not match features"):
            run_step3_scan(
                horde,
                state,
                features,
                cumulants,
                features,
            )

    def test_valid_step3_scan_runs_cleanly(self, horde_and_state) -> None:
        horde, state = horde_and_state
        features = jnp.ones((10, 4), dtype=jnp.float32)
        cumulants = jnp.ones((10, 2), dtype=jnp.float32)
        result = run_step3_scan(
            horde,
            state,
            features,
            cumulants,
            features,
        )
        assert result.td_errors.shape == (10, 2)
        assert result.per_demon_metrics.shape == (10, 2, 3)
        assert result.updates_applied.shape == (10,)


# =============================================================================
# Gymnasium learn_from_trajectory bounds and dimensions
# =============================================================================


class TestGymnasiumTrajectoryScanValidation:
    @pytest.fixture
    def learner(self):
        return LinearLearner(optimizer=LMS(step_size=0.01))

    def test_rejects_non_linear_learner_type(self) -> None:
        obs = jnp.zeros((5, 3), dtype=jnp.float32)
        targets = jnp.zeros((5, 1), dtype=jnp.float32)
        with pytest.raises(TypeError, match="learner must be an instance of LinearLearner"):
            learn_from_trajectory(cast(Any, object()), obs, targets)

    def test_rejects_untrusted_observations(self, learner) -> None:
        targets = jnp.zeros((5, 1), dtype=jnp.float32)
        with pytest.raises(TypeError, match="observations must be a trusted array"):
            learn_from_trajectory(learner, cast(Any, [1.0, 2.0]), targets)

    def test_rejects_untrusted_targets(self, learner) -> None:
        obs = jnp.zeros((5, 3), dtype=jnp.float32)
        with pytest.raises(TypeError, match="targets must be a trusted array"):
            learn_from_trajectory(learner, obs, cast(Any, [1.0, 2.0]))

    def test_rejects_1d_observations(self, learner) -> None:
        obs = jnp.zeros((5,), dtype=jnp.float32)
        targets = jnp.zeros((5, 1), dtype=jnp.float32)
        with pytest.raises(ValueError, match="observations must be a 2D array"):
            learn_from_trajectory(learner, obs, targets)

    def test_rejects_3d_targets(self, learner) -> None:
        obs = jnp.zeros((5, 3), dtype=jnp.float32)
        targets = jnp.zeros((5, 1, 1), dtype=jnp.float32)
        with pytest.raises(ValueError, match="targets must be a 1D or 2D array"):
            learn_from_trajectory(learner, obs, targets)

    def test_rejects_zero_sequence_length(self, learner) -> None:
        obs = jnp.zeros((0, 3), dtype=jnp.float32)
        targets = jnp.zeros((0, 1), dtype=jnp.float32)
        with pytest.raises(ValueError, match="observations sequence length must be an integer"):
            learn_from_trajectory(learner, obs, targets)

    def test_rejects_step_mismatch(self, learner) -> None:
        obs = jnp.zeros((5, 3), dtype=jnp.float32)
        targets = jnp.zeros((4, 1), dtype=jnp.float32)
        with pytest.raises(ValueError, match="targets sequence length must match"):
            learn_from_trajectory(learner, obs, targets)

    def test_rejects_non_float32_observations(self, learner) -> None:
        obs = jnp.zeros((5, 3), dtype=jnp.int32)
        targets = jnp.zeros((5, 1), dtype=jnp.float32)
        with pytest.raises(TypeError, match="observations must have dtype float32"):
            learn_from_trajectory(learner, cast(Any, obs), targets)

    def test_rejects_non_float32_targets(self, learner) -> None:
        obs = jnp.zeros((5, 3), dtype=jnp.float32)
        targets = jnp.zeros((5, 1), dtype=jnp.int32)
        with pytest.raises(TypeError, match="targets must have dtype float32"):
            learn_from_trajectory(learner, obs, cast(Any, targets))

    def test_rejects_mismatched_learner_state(self, learner) -> None:
        obs = jnp.zeros((5, 3), dtype=jnp.float32)
        targets = jnp.zeros((5, 1), dtype=jnp.float32)
        state_wrong_dim = learner.init(4)
        with pytest.raises(ValueError, match="learner_state.weights shape"):
            learn_from_trajectory(learner, obs, targets, learner_state=state_wrong_dim)

    def test_rejects_invalid_learner_state_type(self, learner) -> None:
        obs = jnp.zeros((5, 3), dtype=jnp.float32)
        targets = jnp.zeros((5, 1), dtype=jnp.float32)
        with pytest.raises(TypeError, match="learner_state must be an exact LearnerState"):
            learn_from_trajectory(learner, obs, targets, learner_state=cast(Any, object()))

    def test_valid_trajectory_learning_runs_cleanly(self, learner) -> None:
        obs = jnp.ones((8, 3), dtype=jnp.float32)
        targets = jnp.ones((8, 1), dtype=jnp.float32)
        final_state, metrics = learn_from_trajectory(learner, obs, targets)
        assert isinstance(final_state, LearnerState)
        assert metrics.shape == (8, 3)

    def test_learn_from_trajectory_normalized_alias(self, learner) -> None:
        obs = jnp.ones((8, 3), dtype=jnp.float32)
        targets = jnp.ones((8, 1), dtype=jnp.float32)
        final_state, metrics = learn_from_trajectory_normalized(learner, obs, targets)
        assert isinstance(final_state, LearnerState)
        assert metrics.shape == (8, 3)


# =============================================================================
# PrototypeAgent scan and scan_transitions bounds
# =============================================================================


class TestPrototypeAgentScanValidation:
    @pytest.fixture
    def agent_and_state(self):
        config = PrototypeAgentConfig()
        agent = PrototypeAgent(config)
        key = jr.key(123)
        state = agent.init(key)
        state = agent.start(state, jnp.zeros(config.oak.observation_dim, dtype=jnp.float32))
        return agent, state

    def test_scan_rejects_non_prototype_state(self, agent_and_state) -> None:
        agent, _ = agent_and_state
        rewards = jnp.zeros((5,), dtype=jnp.float32)
        next_obs = jnp.zeros((5, agent.config.oak.observation_dim), dtype=jnp.float32)
        with pytest.raises(TypeError, match="state must be an exact PrototypeAgentState"):
            agent.scan(cast(Any, object()), rewards, next_obs)

    def test_scan_rejects_untrusted_rewards(self, agent_and_state) -> None:
        agent, state = agent_and_state
        next_obs = jnp.zeros((5, agent.config.oak.observation_dim), dtype=jnp.float32)
        with pytest.raises(TypeError, match="rewards must be a trusted array"):
            agent.scan(state, cast(Any, [1.0, 2.0]), next_obs)

    def test_scan_rejects_zero_steps(self, agent_and_state) -> None:
        agent, state = agent_and_state
        rewards = jnp.zeros((0,), dtype=jnp.float32)
        next_obs = jnp.zeros((0, agent.config.oak.observation_dim), dtype=jnp.float32)
        with pytest.raises(
            ValueError, match="rewards must contain between 1 and signed-int32 steps"
        ):
            agent.scan(state, rewards, next_obs)

    def test_scan_transitions_rejects_non_prototype_state(self, agent_and_state) -> None:
        agent, _ = agent_and_state
        obs_dim = agent.config.oak.observation_dim
        transition = PrototypeTransition(
            observation=jnp.zeros((4, obs_dim), dtype=jnp.float32),
            action=jnp.zeros((4,), dtype=jnp.int32),
            decision_id=jnp.zeros((4,), dtype=jnp.int32),
            reward=jnp.zeros((4,), dtype=jnp.float32),
            discount=jnp.ones((4,), dtype=jnp.float32),
            terminated=jnp.zeros((4,), dtype=jnp.bool_),
            truncated=jnp.zeros((4,), dtype=jnp.bool_),
            next_observation=jnp.zeros((4, obs_dim), dtype=jnp.float32),
            next_decision_observation=jnp.zeros((4, obs_dim), dtype=jnp.float32),
        )
        with pytest.raises(TypeError, match="state must be an exact PrototypeAgentState"):
            agent.scan_transitions(cast(Any, object()), transition)

    def test_scan_transitions_rejects_non_transition_type(self, agent_and_state) -> None:
        agent, state = agent_and_state
        with pytest.raises(TypeError, match="transitions must be an exact PrototypeTransition"):
            agent.scan_transitions(state, cast(Any, object()))

    def test_scan_transitions_rejects_zero_steps(self, agent_and_state) -> None:
        agent, state = agent_and_state
        obs_dim = agent.config.oak.observation_dim
        transition = PrototypeTransition(
            observation=jnp.zeros((0, obs_dim), dtype=jnp.float32),
            action=jnp.zeros((0,), dtype=jnp.int32),
            decision_id=jnp.zeros((0,), dtype=jnp.int32),
            reward=jnp.zeros((0,), dtype=jnp.float32),
            discount=jnp.ones((0,), dtype=jnp.float32),
            terminated=jnp.zeros((0,), dtype=jnp.bool_),
            truncated=jnp.zeros((0,), dtype=jnp.bool_),
            next_observation=jnp.zeros((0, obs_dim), dtype=jnp.float32),
            next_decision_observation=jnp.zeros((0, obs_dim), dtype=jnp.float32),
        )
        with pytest.raises(
            ValueError, match="transitions must contain between 1 and signed-int32 steps"
        ):
            agent.scan_transitions(state, transition)

    def test_valid_scan_runs_cleanly(self, agent_and_state) -> None:
        agent, state = agent_and_state
        rewards = jnp.zeros((4,), dtype=jnp.float32)
        next_obs = jnp.zeros((4, agent.config.oak.observation_dim), dtype=jnp.float32)
        result = agent.scan(state, rewards, next_obs)
        assert result.actions.shape == (4,)
        assert result.oak_td_errors.shape == (4,)
