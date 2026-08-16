"""Tests for the SARSAAgent, learning loops, and integration with Horde."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import (
    Autostep,
    DemonType,
    GVFSpec,
    MultiHeadMLPLearner,
    ObGDBounding,
    SARSAAgent,
    SARSAArrayResult,
    SARSAConfig,
    SARSAUpdateResult,
    run_sarsa_from_arrays,
)


def _make_agent(
    n_actions: int = 2,
    hidden_sizes: tuple[int, ...] = (16,),
    gamma: float = 0.99,
    epsilon_start: float = 0.1,
    **kwargs,
) -> SARSAAgent:
    """Helper to create a simple SARSA agent for tests."""
    config = SARSAConfig(
        n_actions=n_actions,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=kwargs.pop("epsilon_end", 0.01),
        epsilon_decay_steps=kwargs.pop("epsilon_decay_steps", 0),
    )
    return SARSAAgent(
        sarsa_config=config,
        hidden_sizes=hidden_sizes,
        sparsity=0.0,
        **kwargs,
    )


# =============================================================================
# Init tests
# =============================================================================


class TestSARSAConfigValidation:
    @pytest.mark.parametrize("n_actions", [0, -1, True, 2.0])
    def test_rejects_invalid_action_count(self, n_actions):
        with pytest.raises(ValueError, match="n_actions"):
            SARSAConfig(n_actions=n_actions)  # type: ignore[arg-type]

    @pytest.mark.parametrize("epsilon_decay_steps", [-1, True, 2.0])
    def test_rejects_invalid_decay_steps(self, epsilon_decay_steps):
        with pytest.raises(ValueError, match="epsilon_decay_steps"):
            SARSAConfig(n_actions=2, epsilon_decay_steps=epsilon_decay_steps)  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["gamma", "epsilon_start", "epsilon_end"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
    def test_rejects_invalid_probability(self, name, value):
        with pytest.raises(ValueError, match=name):
            SARSAConfig(n_actions=2, **{name: value})

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_accepts_probability_endpoints(self, value):
        config = SARSAConfig(
            n_actions=2,
            gamma=value,
            epsilon_start=value,
            epsilon_end=value,
        )
        assert config.gamma == value

    def test_rejects_class_spoofed_scalars(self):
        class SpoofedInt:
            @property
            def __class__(self):
                return int

            def __int__(self):
                return 2

        class SpoofedFloat:
            @property
            def __class__(self):
                return float

            def __float__(self):
                return 0.5

        with pytest.raises(ValueError, match="n_actions"):
            SARSAConfig(n_actions=SpoofedInt())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="gamma"):
            SARSAConfig(n_actions=2, gamma=SpoofedFloat())  # type: ignore[arg-type]

    def test_accepts_gamma_one_and_canonicalizes_numpy_scalars(self) -> None:
        config = SARSAConfig(
            n_actions=2,
            gamma=np.float32(1.0),
            epsilon_start=np.float32(0.25),
            epsilon_end=np.float64(0.1),
        )

        assert config.gamma == 1.0
        assert type(config.gamma) is float
        assert type(config.epsilon_start) is float
        assert type(config.epsilon_end) is float

    def test_rejects_inverted_decay_schedule(self) -> None:
        with pytest.raises(ValueError, match="epsilon_end"):
            SARSAConfig(
                n_actions=2,
                epsilon_start=0.3,
                epsilon_end=0.5,
                epsilon_decay_steps=10,
            )


class TestSARSAInit:
    """Tests for SARSAAgent initialization."""

    def test_init_shapes(self):
        """State arrays have correct shapes."""
        agent = _make_agent(n_actions=3)
        state = agent.init(feature_dim=5, key=jr.key(42))

        chex.assert_shape(state.last_action, ())
        chex.assert_shape(state.last_observation, (5,))
        chex.assert_shape(state.epsilon, ())
        chex.assert_shape(state.step_count, ())
        assert state.step_count == 0
        assert state.last_action == -1

    def test_q_value_prediction(self):
        """Q-values have shape (n_actions,)."""
        agent = _make_agent(n_actions=4)
        state = agent.init(feature_dim=5, key=jr.key(42))
        obs = jnp.ones(5, dtype=jnp.float32)

        all_preds = agent.horde.predict(state.learner_state, obs)
        q_values = all_preds[: agent.n_actions]
        chex.assert_shape(q_values, (4,))

    def test_n_demons_matches_n_actions(self):
        """Horde has exactly n_actions demons (no prediction demons)."""
        agent = _make_agent(n_actions=3)
        assert agent.horde.n_demons == 3

    def test_with_prediction_demons(self):
        """Prediction demons are appended after control demons."""
        pred_demons = [
            GVFSpec(
                name="pred_0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),
        ]
        agent = _make_agent(n_actions=2, prediction_demons=pred_demons)
        assert agent.horde.n_demons == 3
        assert agent.horde.horde_spec.demons[0].demon_type == DemonType.CONTROL
        assert agent.horde.horde_spec.demons[1].demon_type == DemonType.CONTROL
        assert agent.horde.horde_spec.demons[2].demon_type == DemonType.PREDICTION


# =============================================================================
# Action selection tests
# =============================================================================


class TestSARSAActionSelection:
    """Tests for epsilon-greedy action selection."""

    def test_greedy_when_epsilon_zero(self):
        """With epsilon=0, always selects greedy action."""
        agent = _make_agent(n_actions=3, epsilon_start=0.0)
        state = agent.init(feature_dim=5, key=jr.key(42))
        obs = jnp.ones(5, dtype=jnp.float32)

        # Run many selections — should always be greedy
        actions = []
        for i in range(50):
            action, new_key = agent.select_action(state, obs)
            state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]
            actions.append(int(action))

        # All actions should be the same (greedy)
        assert len(set(actions)) == 1

    def test_random_when_epsilon_one(self):
        """With epsilon=1, always explores (random actions)."""
        agent = _make_agent(n_actions=4, epsilon_start=1.0)
        state = agent.init(feature_dim=5, key=jr.key(42))
        # Override epsilon in state
        state = state.replace(epsilon=jnp.array(1.0))  # type: ignore[attr-defined]
        obs = jnp.ones(5, dtype=jnp.float32)

        actions = []
        for _ in range(200):
            action, new_key = agent.select_action(state, obs)
            state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]
            actions.append(int(action))

        # With 200 samples and 4 actions, we should see multiple distinct actions
        unique_actions = set(actions)
        assert len(unique_actions) >= 2, f"Expected multiple actions, got {unique_actions}"

    def test_tie_breaking_uniform(self):
        """Equal Q-values should produce roughly uniform action selection.

        Gumbel trick tie-breaking should avoid left-side bias from jnp.argmax.
        """
        agent = _make_agent(n_actions=4, epsilon_start=0.0)
        state = agent.init(feature_dim=5, key=jr.key(42))
        obs = jnp.zeros(5, dtype=jnp.float32)  # zero obs -> similar Q-values

        counts = np.zeros(4)
        n_samples = 2000
        for _ in range(n_samples):
            action, new_key = agent.select_action(state, obs)
            state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]
            counts[int(action)] += 1

        # Chi-squared test for uniformity
        expected = n_samples / 4
        chi_sq = np.sum((counts - expected) ** 2 / expected)
        # With df=3, chi_sq < 16.27 at p=0.001 (very conservative)
        assert chi_sq < 16.27, f"Action distribution not uniform: {counts}, chi_sq={chi_sq}"

    def test_action_in_valid_range(self):
        """Selected actions are always in [0, n_actions)."""
        agent = _make_agent(n_actions=6, epsilon_start=0.5)
        state = agent.init(feature_dim=5, key=jr.key(0))
        obs = jnp.ones(5, dtype=jnp.float32)

        for _ in range(100):
            action, new_key = agent.select_action(state, obs)
            state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]
            assert 0 <= int(action) < 6


# =============================================================================
# Update tests
# =============================================================================


class TestSARSAUpdate:
    """Tests for SARSA update logic."""

    def test_sarsa_target(self):
        """SARSA target is r + gamma * Q(s', a')."""
        agent = _make_agent(n_actions=2, gamma=0.9, epsilon_start=0.0)
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        # Set up last_action and last_observation
        action, new_key = agent.select_action(state, obs)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=action,
            last_observation=obs,
            rng_key=new_key,
        )

        next_obs = jnp.ones(4, dtype=jnp.float32) * 2.0
        next_action, new_key = agent.select_action(state, next_obs)
        state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]

        result = agent.update(
            state,
            reward=jnp.array(1.0),
            observation=next_obs,
            terminated=jnp.array(0.0),
            next_action=next_action,
        )

        assert isinstance(result, SARSAUpdateResult)
        chex.assert_shape(result.q_values, (2,))
        chex.assert_shape(result.td_error, ())
        assert result.reward == 1.0

    def test_update_before_select_action_trains_no_head(self):
        """last_action == -1 must not wrap to the last control head."""
        agent = _make_agent(n_actions=4, epsilon_start=0.0)
        state = agent.init(feature_dim=4, key=jr.key(3))
        next_obs = jnp.ones(4, dtype=jnp.float32) * 2.0

        # No select_action was called, so last_action == -1.
        assert int(state.last_action) == -1
        result = agent.update(
            state,
            reward=jnp.array(1.0),
            observation=next_obs,
            terminated=jnp.array(0.0),
            next_action=jnp.array(0, dtype=jnp.int32),
        )

        assert float(result.td_error) == 0.0
        chex.assert_trees_all_close(
            result.state.learner_state.trunk_params.weights,
            state.learner_state.trunk_params.weights,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            result.state.learner_state.trunk_params.biases,
            state.learner_state.trunk_params.biases,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            result.state.learner_state.head_params.weights,
            state.learner_state.head_params.weights,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            result.state.learner_state.head_params.biases,
            state.learner_state.head_params.biases,
            atol=0.0,
        )

    def test_update_before_select_action_is_exact_noop_with_nonzero_traces(self):
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(),
            gamma=0.9,
            epsilon_start=0.2,
            epsilon_decay_steps=10,
            lamda=0.8,
        )
        state = agent.init(feature_dim=3, key=jr.key(9))
        learner_state = state.learner_state.replace(
            head_traces=jax.tree.map(jnp.ones_like, state.learner_state.head_traces)
        )
        state = state.replace(learner_state=learner_state)

        result = agent.update(
            state,
            reward=jnp.array(1.0),
            observation=jnp.ones(3),
            terminated=jnp.array(0.0),
            next_action=jnp.array(0, dtype=jnp.int32),
        )

        # birth_timestamp is host-only static PyTree metadata and is not a
        # meaningful transaction field inside a jitted comparison.
        actual = result.state.replace(
            learner_state=result.state.learner_state.replace(birth_timestamp=0.0)
        )
        expected = state.replace(learner_state=state.learner_state.replace(birth_timestamp=0.0))
        chex.assert_trees_all_equal(actual, expected)
        assert float(result.td_error) == 0.0

    def test_terminated_no_bootstrap(self):
        """At terminal state, target = r (no bootstrapping)."""
        agent = _make_agent(n_actions=2, gamma=0.99, epsilon_start=0.0)
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        action, new_key = agent.select_action(state, obs)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=action,
            last_observation=obs,
            rng_key=new_key,
        )

        next_obs = jnp.zeros(4, dtype=jnp.float32)
        next_action = jnp.array(0, dtype=jnp.int32)

        # Non-terminal: target = r + gamma * Q(s', a')
        result_nt = agent.update(
            state,
            reward=jnp.array(1.0),
            observation=next_obs,
            terminated=jnp.array(0.0),
            next_action=next_action,
        )

        # Terminal: target = r
        result_t = agent.update(
            state,
            reward=jnp.array(1.0),
            observation=next_obs,
            terminated=jnp.array(1.0),
            next_action=next_action,
        )

        # TD errors should differ (unless Q(s', a') happens to be exactly 0)
        # At minimum, the logic should run without error
        assert not jnp.isnan(result_nt.td_error)
        assert not jnp.isnan(result_t.td_error)

    def test_terminated_inf_next_q_is_not_nan_td_error(self) -> None:
        """Terminal 0 * inf Q(s', a') is NaN and poisons the SARSA target.

        Fail-closed: a terminal transition does not multiply Q(s', a').
        The target is the reward, matching the finite-path algebra.
        """
        agent = _make_agent(n_actions=2, gamma=0.99, epsilon_start=0.0, hidden_sizes=())
        state = agent.init(feature_dim=2, key=jr.key(0))
        obs = jnp.ones(2, dtype=jnp.float32)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )
        result = agent.update(
            state,
            reward=jnp.array(1.0, dtype=jnp.float32),
            observation=jnp.array([jnp.inf, 0.0], dtype=jnp.float32),
            terminated=jnp.array(True),
            next_action=jnp.array(0, dtype=jnp.int32),
        )
        assert bool(jnp.isfinite(result.td_error))
        q_old = agent.horde.predict(state.learner_state, obs)[0]
        chex.assert_trees_all_close(result.td_error, jnp.float32(1.0) - q_old)

    def test_zero_gamma_inf_next_q_is_not_nan_td_error(self) -> None:
        """Continuing 0 * inf Q(s', a') is NaN when gamma is exactly 0.

        jnp.where still evaluates both branches, so skip the product when
        gamma is 0 rather than selecting after 0 * inf.
        """
        agent = _make_agent(n_actions=2, gamma=0.0, epsilon_start=0.0, hidden_sizes=())
        state = agent.init(feature_dim=2, key=jr.key(0))
        obs = jnp.ones(2, dtype=jnp.float32)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )
        result = agent.update(
            state,
            reward=jnp.array(1.0, dtype=jnp.float32),
            observation=jnp.array([jnp.inf, 0.0], dtype=jnp.float32),
            terminated=jnp.array(False),
            next_action=jnp.array(0, dtype=jnp.int32),
        )
        assert bool(jnp.isfinite(result.td_error))
        q_old = agent.horde.predict(state.learner_state, obs)[0]
        chex.assert_trees_all_close(result.td_error, jnp.float32(1.0) - q_old)

    def test_zero_gamma_does_not_decay_inf_inactive_traces(self) -> None:
        """Inactive-head decay is gamma*lamda; 0 * inf traces is NaN."""
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(),
            gamma=0.0,
            epsilon_start=0.0,
            lamda=0.8,
        )
        state = agent.init(feature_dim=2, key=jr.key(1))
        obs = jnp.ones(2, dtype=jnp.float32)
        inf_w = jnp.full_like(state.learner_state.head_traces[1][0], jnp.inf)
        inf_b = jnp.full_like(state.learner_state.head_traces[1][1], jnp.inf)
        traces = list(state.learner_state.head_traces)
        traces[1] = (inf_w, inf_b)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
            learner_state=state.learner_state.replace(head_traces=tuple(traces)),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))
        result = agent.update(
            state,
            reward=jnp.array(0.5, dtype=jnp.float32),
            observation=obs * 0.5,
            terminated=jnp.array(False),
            next_action=jnp.array(0, dtype=jnp.int32),
        )
        w_trace, b_trace = result.state.learner_state.head_traces[1]
        assert bool(jnp.all(jnp.isfinite(w_trace)))
        assert bool(jnp.all(jnp.isfinite(b_trace)))
        chex.assert_trees_all_close(w_trace, jnp.zeros_like(w_trace))
        chex.assert_trees_all_close(b_trace, jnp.zeros_like(b_trace))

    def test_nan_masking(self):
        """Only the taken action's head receives a weight update."""
        agent = _make_agent(n_actions=3, gamma=0.9, epsilon_start=0.0)
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        # Force action = 1
        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(1, dtype=jnp.int32),
            last_observation=obs,
        )

        next_obs = jnp.ones(4, dtype=jnp.float32) * 0.5
        next_action = jnp.array(0, dtype=jnp.int32)

        # Save head params before update
        old_head_weights = [state.learner_state.head_params.weights[i] for i in range(3)]

        result = agent.update(
            state,
            reward=jnp.array(1.0),
            observation=next_obs,
            terminated=jnp.array(0.0),
            next_action=next_action,
        )

        new_head_weights = [result.state.learner_state.head_params.weights[i] for i in range(3)]

        # Head 1 should have changed (it was the taken action)
        head1_changed = not jnp.allclose(old_head_weights[1], new_head_weights[1])
        assert head1_changed, "Head 1 (taken action) should have been updated"

        # Heads 0 and 2 should be unchanged (NaN targets)
        chex.assert_trees_all_close(old_head_weights[0], new_head_weights[0])
        chex.assert_trees_all_close(old_head_weights[2], new_head_weights[2])

    def test_prediction_demons_unaffected(self):
        """Prediction demons learn alongside Q-heads without interference."""
        pred_demons = [
            GVFSpec(
                name="pred_0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),
        ]
        agent = _make_agent(n_actions=2, prediction_demons=pred_demons)
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )

        next_obs = jnp.ones(4, dtype=jnp.float32) * 0.5

        # Prediction cumulant for the prediction demon
        pred_cumulants = jnp.array([2.0], dtype=jnp.float32)
        next_action = jnp.array(1, dtype=jnp.int32)

        old_pred_weights = state.learner_state.head_params.weights[2]

        result = agent.update(
            state,
            reward=jnp.array(1.0),
            observation=next_obs,
            terminated=jnp.array(0.0),
            next_action=next_action,
            prediction_cumulants=pred_cumulants,
        )

        new_pred_weights = result.state.learner_state.head_params.weights[2]
        pred_changed = not jnp.allclose(old_pred_weights, new_pred_weights)
        assert pred_changed, "Prediction demon head should have been updated"

    def test_td_error_uses_last_observation_prediction(self):
        """Returned TD error is target - Q(s_t, a_t), not Q(s_{t+1}, a_t)."""
        agent = _make_agent(n_actions=2, hidden_sizes=(), gamma=0.0, epsilon_start=0.0)
        state = agent.init(feature_dim=2, key=jr.key(42))
        head_weights = state.learner_state.head_params.weights
        state = state.replace(  # type: ignore[attr-defined]
            learner_state=state.learner_state.replace(  # type: ignore[attr-defined]
                head_params=state.learner_state.head_params.replace(  # type: ignore[attr-defined]
                    weights=(
                        head_weights[0].at[0, 0].set(2.0),
                        head_weights[1],
                    )
                )
            ),
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        )

        next_observation = jnp.array([0.0, 1.0], dtype=jnp.float32)
        reward = jnp.array(1.0, dtype=jnp.float32)
        result = agent.update(
            state,
            reward=reward,
            observation=next_observation,
            terminated=jnp.array(0.0, dtype=jnp.float32),
            next_action=jnp.array(1, dtype=jnp.int32),
        )

        previous_q = agent.horde.predict(
            state.learner_state,
            state.last_observation,
        )[0]
        next_same_action_q = agent.horde.predict(
            state.learner_state,
            next_observation,
        )[0]
        chex.assert_trees_all_close(result.td_error, reward - previous_q)
        assert not jnp.allclose(result.td_error, reward - next_same_action_q)

    def test_sarsa_vs_qlearning_different_targets(self):
        """SARSA uses Q(s', a') while Q-learning uses max Q(s', :)."""
        agent = _make_agent(n_actions=3, gamma=0.9, epsilon_start=0.0)
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )

        next_obs = jnp.ones(4, dtype=jnp.float32) * 2.0
        reward = jnp.array(1.0)

        # Get Q(s', :)
        all_preds = agent.horde.predict(state.learner_state, next_obs)
        q_next = all_preds[: agent.n_actions]

        # SARSA target with a' = action 0
        sarsa_target_a0 = reward + 0.9 * q_next[0]
        # Q-learning target
        qlearning_target = reward + 0.9 * jnp.max(q_next)

        # Unless all Q-values are equal, SARSA targets differ from Q-learning
        if not jnp.allclose(q_next[0], jnp.max(q_next)):
            assert not jnp.allclose(sarsa_target_a0, qlearning_target)


# =============================================================================
# Epsilon decay tests
# =============================================================================


class TestSARSAEpsilonDecay:
    """Tests for epsilon scheduling."""

    def test_linear_decay(self):
        """Epsilon decays linearly from start to end over N steps."""
        agent = _make_agent(
            n_actions=2,
            epsilon_start=1.0,
            epsilon_end=0.0,
            epsilon_decay_steps=100,
        )
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )

        next_obs = obs
        next_action = jnp.array(0, dtype=jnp.int32)

        # Run 50 steps
        for _ in range(50):
            result = agent.update(
                state,
                reward=jnp.array(0.0),
                observation=next_obs,
                terminated=jnp.array(0.0),
                next_action=next_action,
            )
            state = result.state

        # After 50 steps: epsilon should be ~0.5
        assert abs(float(state.epsilon) - 0.5) < 0.02

    def test_no_decay_when_zero_steps(self):
        """Epsilon stays constant when decay_steps=0."""
        agent = _make_agent(
            n_actions=2,
            epsilon_start=0.5,
            epsilon_decay_steps=0,
        )
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )

        for _ in range(20):
            result = agent.update(
                state,
                reward=jnp.array(0.0),
                observation=obs,
                terminated=jnp.array(0.0),
                next_action=jnp.array(0, dtype=jnp.int32),
            )
            state = result.state

        assert float(state.epsilon) == 0.5

    def test_epsilon_floors_at_end(self):
        """Epsilon doesn't go below epsilon_end."""
        agent = _make_agent(
            n_actions=2,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=10,
        )
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )

        for _ in range(100):
            result = agent.update(
                state,
                reward=jnp.array(0.0),
                observation=obs,
                terminated=jnp.array(0.0),
                next_action=jnp.array(0, dtype=jnp.int32),
            )
            state = result.state

        assert float(state.epsilon) >= 0.1 - 1e-6


# =============================================================================
# Gymnasium integration tests
# =============================================================================


class TestSARSAGymnasium:
    """Tests for SARSA with Gymnasium environments."""

    def test_cartpole_no_crash(self):
        """Run 100 steps on CartPole without crashing."""
        gymnasium = pytest.importorskip("gymnasium")
        gym = gymnasium

        env = gym.make("CartPole-v1")
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(16,),
            gamma=0.99,
            epsilon_start=0.5,
        )
        state = agent.init(feature_dim=4, key=jr.key(42))

        from alberta_framework import run_sarsa_continuing

        result = run_sarsa_continuing(agent, state, env, num_steps=100)

        assert len(result.rewards) == 100
        assert not any(np.isnan(r) for r in result.rewards)
        env.close()

    def test_episode_mode(self):
        """Run one episode on CartPole."""
        gymnasium = pytest.importorskip("gymnasium")
        gym = gymnasium

        env = gym.make("CartPole-v1")
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(16,),
            gamma=0.99,
            epsilon_start=0.5,
        )
        state = agent.init(feature_dim=4, key=jr.key(42))

        from alberta_framework import run_sarsa_episode

        result = run_sarsa_episode(agent, state, env, max_steps=500)

        assert result.num_steps > 0
        assert result.num_steps <= 500
        assert len(result.rewards) == result.num_steps
        env.close()


# =============================================================================
# Bounder + optimizer tests
# =============================================================================


class TestSARSAWithBounder:
    """Tests for SARSA with ObGDBounding."""

    def test_obgd_no_divergence(self):
        """SARSA + ObGDBounding doesn't diverge over 50 steps."""
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(16,),
            bounder=ObGDBounding(),
        )
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )

        for _ in range(50):
            next_obs = obs + jax.random.normal(jr.key(0), (4,)) * 0.1
            next_action = jnp.array(0, dtype=jnp.int32)
            result = agent.update(
                state,
                reward=jnp.array(1.0),
                observation=next_obs,
                terminated=jnp.array(0.0),
                next_action=next_action,
            )
            state = result.state

            # Check Q-values are finite
            q_vals = agent.horde.predict(state.learner_state, obs)
            assert jnp.all(jnp.isfinite(q_vals)), f"Q-values diverged: {q_vals}"


class TestSARSAWithAutostep:
    """Tests for SARSA with Autostep optimizer."""

    def test_autostep_runs(self):
        """SARSA + Autostep runs without errors."""
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(16,),
            optimizer=Autostep(),
        )
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )

        for _ in range(20):
            result = agent.update(
                state,
                reward=jnp.array(1.0),
                observation=obs,
                terminated=jnp.array(0.0),
                next_action=jnp.array(1, dtype=jnp.int32),
            )
            state = result.state

        # Should complete without NaN
        q_vals = agent.horde.predict(state.learner_state, obs)
        assert jnp.all(jnp.isfinite(q_vals))


# =============================================================================
# Config serialization tests
# =============================================================================


class TestSARSAConfigSerialization:
    """Tests for SARSA config serialization roundtrip."""

    def test_sarsa_config_roundtrip(self):
        """SARSAConfig serializes and deserializes correctly."""
        config = SARSAConfig(
            n_actions=4,
            gamma=0.95,
            epsilon_start=0.2,
            epsilon_end=0.05,
            epsilon_decay_steps=1000,
        )
        restored = SARSAConfig.from_config(config.to_config())
        assert restored.n_actions == 4
        assert restored.gamma == 0.95
        assert restored.epsilon_start == 0.2
        assert restored.epsilon_end == 0.05
        assert restored.epsilon_decay_steps == 1000

    def test_agent_config_roundtrip(self):
        """SARSAAgent serializes and deserializes correctly."""
        pred_demons = [
            GVFSpec(
                name="pred_0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),
        ]
        agent = _make_agent(
            n_actions=3,
            hidden_sizes=(32, 16),
            gamma=0.95,
            prediction_demons=pred_demons,
        )

        config = agent.to_config()
        restored = SARSAAgent.from_config(config)

        assert restored.n_actions == 3
        assert restored.sarsa_config.gamma == 0.95
        assert restored.horde.n_demons == 4  # 3 control + 1 prediction

    def test_agent_config_roundtrip_no_prediction(self):
        """SARSAAgent without prediction demons roundtrips correctly."""
        agent = _make_agent(n_actions=2)
        config = agent.to_config()
        restored = SARSAAgent.from_config(config)
        assert restored.n_actions == 2
        assert restored.horde.n_demons == 2

    def test_agent_config_rejects_unknown_state_schema(self):
        """Serialized state schemas fail closed when the loader does not support them."""
        config = _make_agent(n_actions=2).to_config()
        config["state_schema"] = "alberta.multi-head-mlp-state.v999"

        with pytest.raises(ValueError, match="unsupported SARSA Horde state schema"):
            SARSAAgent.from_config(config)


# =============================================================================
# Scan-based (array) loop tests
# =============================================================================


class TestSARSAScan:
    """Tests for run_sarsa_from_arrays scan loop."""

    def test_scan_shapes(self):
        """Scan loop produces correct output shapes."""
        agent = _make_agent(n_actions=2, hidden_sizes=(8,))
        state = agent.init(feature_dim=4, key=jr.key(42))

        n_steps = 20
        obs = jax.random.normal(jr.key(0), (n_steps, 4))
        next_obs = jax.random.normal(jr.key(1), (n_steps, 4))
        rewards = jnp.ones(n_steps)
        terminated = jnp.zeros(n_steps)

        # Set initial action/observation
        action, new_key = agent.select_action(state, obs[0])
        state = state.replace(  # type: ignore[attr-defined]
            last_action=action,
            last_observation=obs[0],
            rng_key=new_key,
        )

        result = run_sarsa_from_arrays(agent, state, obs, rewards, terminated, next_obs)

        assert isinstance(result, SARSAArrayResult)
        chex.assert_shape(result.q_values, (n_steps, 2))
        chex.assert_shape(result.td_errors, (n_steps,))
        chex.assert_shape(result.actions, (n_steps,))
        assert jnp.all(jnp.isfinite(result.td_errors))

    def test_scan_terminal_handling(self):
        """Scan loop handles terminal flags correctly."""
        agent = _make_agent(n_actions=2, hidden_sizes=(8,), gamma=0.99)
        state = agent.init(feature_dim=4, key=jr.key(42))

        n_steps = 10
        obs = jnp.ones((n_steps, 4))
        next_obs = jnp.ones((n_steps, 4))
        rewards = jnp.ones(n_steps)
        # Terminal at step 5
        terminated = jnp.zeros(n_steps).at[5].set(1.0)

        action, new_key = agent.select_action(state, obs[0])
        state = state.replace(  # type: ignore[attr-defined]
            last_action=action,
            last_observation=obs[0],
            rng_key=new_key,
        )

        result = run_sarsa_from_arrays(agent, state, obs, rewards, terminated, next_obs)

        # Should run without error and produce finite results
        assert jnp.all(jnp.isfinite(result.td_errors))


# =============================================================================
# Trunk trace guard tests (Phase 1)
# =============================================================================


class TestTrunkTraceGuard:
    """Tests for trunk gamma*lambda validation."""

    def test_trunk_trace_decay_raises(self):
        """MultiHeadMLPLearner with trunk gamma*lamda>0 and hidden layers raises."""
        import pytest

        with pytest.raises(ValueError, match="Trunk gamma\\*lamda must be 0"):
            MultiHeadMLPLearner(
                n_heads=2,
                hidden_sizes=(16,),
                gamma=0.9,
                lamda=0.5,
            )

    def test_trunk_trace_decay_allowed_linear(self):
        """Linear baseline (hidden_sizes=()) allows any gamma*lamda."""
        # Should NOT raise
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(),
            gamma=0.9,
            lamda=0.5,
        )
        state = learner.init(feature_dim=4, key=jr.key(42))
        assert state is not None

    def test_trunk_gamma_zero_ok(self):
        """gamma=0 with any lamda is fine for MLP."""
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(16,),
            gamma=0.0,
            lamda=0.9,
        )
        state = learner.init(feature_dim=4, key=jr.key(42))
        assert state is not None

    def test_trunk_lamda_zero_ok(self):
        """lamda=0 with any gamma is fine for MLP."""
        learner = MultiHeadMLPLearner(
            n_heads=2,
            hidden_sizes=(16,),
            gamma=0.99,
            lamda=0.0,
        )
        state = learner.init(feature_dim=4, key=jr.key(42))
        assert state is not None


# =============================================================================
# SARSA(lambda) eligibility trace tests
# =============================================================================


class TestSARSALambdaTraces:
    """Tests for SARSA(lambda) eligibility-trace decay on control heads.

    Control demons carry the real discount so head traces decay by
    ``config.gamma * lamda`` — with the TD target still computed
    externally (no internal bootstrap).
    """

    def test_control_head_trace_decay_factor(self):
        """Active control head's trace decays by gamma*lamda, not 0."""
        gamma, lamda = 0.9, 0.8
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(),
            gamma=gamma,
            epsilon_start=0.0,
            lamda=lamda,
        )
        state = agent.init(feature_dim=3, key=jr.key(0))
        obs1 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        obs2 = jnp.array([-1.0, 0.5, 2.0], dtype=jnp.float32)
        obs3 = jnp.array([0.2, -0.4, 1.0], dtype=jnp.float32)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs1,
        )

        stay = jnp.array(0, dtype=jnp.int32)
        result = agent.update(state, jnp.array(0.5), obs2, jnp.array(0.0), stay)
        result = agent.update(result.state, jnp.array(0.3), obs3, jnp.array(0.0), stay)

        # Linear model: grad of Q(s, 0) wrt head-0 weights is s itself, so
        # after two visits e_2 = (gamma * lamda) * s_1 + s_2.
        gl = gamma * lamda
        w_trace, b_trace = result.state.learner_state.head_traces[0]
        chex.assert_trees_all_close(w_trace, (gl * obs1 + obs2).reshape(1, -1), rtol=1e-5)
        chex.assert_trees_all_close(b_trace, jnp.array([gl + 1.0], dtype=jnp.float32), rtol=1e-5)

    def test_lambda_changes_multistep_updates(self):
        """lamda in {0.0, 0.9} produce different weights on a 3-step corridor."""

        def run(lamda: float):
            agent = _make_agent(
                n_actions=1,
                hidden_sizes=(),
                gamma=0.9,
                epsilon_start=0.0,
                lamda=lamda,
            )
            state = agent.init(feature_dim=3, key=jr.key(7))
            # Deterministic corridor s0 -> s1 -> s2 -> terminal, reward at end
            obs = jnp.eye(3, dtype=jnp.float32)
            next_obs = jnp.roll(obs, -1, axis=0)
            rewards = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
            terminated = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
            state = state.replace(  # type: ignore[attr-defined]
                last_action=jnp.array(0, dtype=jnp.int32),
                last_observation=obs[0],
            )
            result = run_sarsa_from_arrays(agent, state, obs, rewards, terminated, next_obs)
            return result.state.learner_state.head_params.weights[0]

        w_one_step = run(0.0)
        w_trace = run(0.9)
        assert not jnp.allclose(w_one_step, w_trace), (
            "SARSA(lambda) must assign multi-step credit differently from "
            "one-step SARSA, but lamda had no effect on the updates"
        )

    def test_inactive_control_head_trace_decays(self):
        """A head not matching the taken action decays its trace each step."""
        gamma, lamda = 0.9, 0.8
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(),
            gamma=gamma,
            epsilon_start=0.0,
            lamda=lamda,
        )
        state = agent.init(feature_dim=3, key=jr.key(3))
        obs1 = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
        obs2 = jnp.array([0.3, 1.0, -1.0], dtype=jnp.float32)
        obs3 = jnp.array([-0.5, 0.2, 2.0], dtype=jnp.float32)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs1,
        )

        switch = jnp.array(1, dtype=jnp.int32)
        # Step 1: head 0 active (trace = obs1); next action switches to 1
        result = agent.update(state, jnp.array(0.0), obs2, jnp.array(0.0), switch)
        # Step 2: head 1 active; head 0 must decay by gamma*lamda, not freeze
        result = agent.update(result.state, jnp.array(0.0), obs3, jnp.array(0.0), switch)

        gl = gamma * lamda
        w_trace, b_trace = result.state.learner_state.head_traces[0]
        chex.assert_trees_all_close(w_trace, (gl * obs1).reshape(1, -1), rtol=1e-5)
        chex.assert_trees_all_close(b_trace, jnp.array([gl], dtype=jnp.float32), rtol=1e-5)

    def test_traces_reset_at_episode_boundary(self):
        """Control-head traces are cleared after a terminated transition."""
        agent = _make_agent(
            n_actions=2,
            hidden_sizes=(),
            gamma=0.9,
            epsilon_start=0.0,
            lamda=0.8,
        )
        state = agent.init(feature_dim=3, key=jr.key(1))
        obs1 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        obs2 = jnp.array([-1.0, 0.5, 2.0], dtype=jnp.float32)
        obs3 = jnp.zeros(3, dtype=jnp.float32)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs1,
        )

        stay = jnp.array(0, dtype=jnp.int32)
        # Build up a trace, then hit an episode boundary
        result = agent.update(state, jnp.array(0.0), obs2, jnp.array(0.0), stay)
        result = agent.update(result.state, jnp.array(1.0), obs3, jnp.array(1.0), stay)

        for i in range(2):
            w_trace, b_trace = result.state.learner_state.head_traces[i]
            assert jnp.allclose(w_trace, 0.0), f"head {i} w-trace not reset"
            assert jnp.allclose(b_trace, 0.0), f"head {i} b-trace not reset"

    def test_external_target_semantics_intact(self):
        """With lamda>0, the TD target is still r + gamma*Q(s',a') (no
        double-counted internal bootstrap)."""
        agent = _make_agent(n_actions=2, hidden_sizes=(), gamma=0.9, epsilon_start=0.0, lamda=0.8)
        state = agent.init(feature_dim=3, key=jr.key(5))
        obs1 = jnp.array([1.0, -1.0, 0.5], dtype=jnp.float32)
        obs2 = jnp.array([0.5, 2.0, -0.3], dtype=jnp.float32)
        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs1,
        )

        next_action = jnp.array(1, dtype=jnp.int32)
        reward = jnp.array(1.0, dtype=jnp.float32)
        q_next = agent.horde.predict(state.learner_state, obs2)[:2]
        q_old = agent.horde.predict(state.learner_state, obs1)[0]
        expected_td = reward + 0.9 * q_next[next_action] - q_old

        result = agent.update(state, reward, obs2, jnp.array(0.0), next_action)
        chex.assert_trees_all_close(result.td_error, expected_td, rtol=1e-5)


# =============================================================================
# Continuing pseudo-boundary test
# =============================================================================


class TestSARSAContinuingPseudoBoundary:
    """Tests for continuing mode at pseudo-boundaries."""

    def test_pseudo_boundary_zeros_gamma(self):
        """At a pseudo-boundary (episode end), gamma=0 prevents bootstrapping."""
        agent = _make_agent(n_actions=2, gamma=0.99, epsilon_start=0.0)
        state = agent.init(feature_dim=4, key=jr.key(42))
        obs = jnp.ones(4, dtype=jnp.float32)

        state = state.replace(  # type: ignore[attr-defined]
            last_action=jnp.array(0, dtype=jnp.int32),
            last_observation=obs,
        )

        next_obs = jnp.ones(4, dtype=jnp.float32) * 3.0
        next_action = jnp.array(1, dtype=jnp.int32)
        reward = jnp.array(5.0)

        # Terminal update: target = r only
        result_term = agent.update(state, reward, next_obs, jnp.array(1.0), next_action)

        # Non-terminal update: target = r + gamma * Q(s', a')
        result_cont = agent.update(state, reward, next_obs, jnp.array(0.0), next_action)

        # TD errors should differ (terminal strips bootstrap)
        # Both should be finite
        assert jnp.isfinite(result_term.td_error)
        assert jnp.isfinite(result_cont.td_error)


def test_sarsa_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        SARSAConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        SARSAConfig(n_actions=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon_decay_steps"):
        SARSAConfig(n_actions=2, epsilon_decay_steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon_decay_steps"):
        SARSAConfig(n_actions=2, epsilon_decay_steps=100.5)  # type: ignore[arg-type]


def test_sarsa_config_accepts_and_canonicalizes_numpy_integers() -> None:
    cfg = SARSAConfig(
        n_actions=np.int32(4),
        epsilon_decay_steps=np.int64(100),
    )
    assert type(cfg.n_actions) is int
    assert type(cfg.epsilon_decay_steps) is int
    assert cfg.n_actions == 4
    assert cfg.epsilon_decay_steps == 100


@pytest.mark.parametrize(
    "integer_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ],
)
def test_sarsa_config_accepts_full_numpy_integer_family(integer_type) -> None:
    config = SARSAConfig(
        n_actions=integer_type(2),
        epsilon_decay_steps=integer_type(3),
    )
    assert type(config.n_actions) is int
    assert type(config.epsilon_decay_steps) is int


def test_sarsa_config_rejects_hostile_integer_hook_without_calling_it() -> None:
    class HostileIndex:
        def __index__(self) -> int:
            raise AssertionError("untrusted __index__ must not run")

        def __repr__(self) -> str:
            raise AssertionError("untrusted __repr__ must not run")

    with pytest.raises(ValueError, match="n_actions"):
        SARSAConfig(n_actions=HostileIndex())  # type: ignore[arg-type]


@pytest.mark.parametrize("float_type", [np.dtype(code).type for code in ("e", "f", "d", "g")])
def test_sarsa_config_accepts_full_numpy_float_family(float_type) -> None:
    config = SARSAConfig(
        n_actions=2,
        gamma=float_type(0.5),
        epsilon_start=float_type(0.25),
        epsilon_end=float_type(0.125),
    )
    assert type(config.gamma) is float
    assert type(config.epsilon_start) is float
    assert type(config.epsilon_end) is float


@pytest.mark.parametrize("field", ["gamma", "epsilon_start", "epsilon_end"])
def test_sarsa_config_rejects_float_subclasses_without_calling_hooks(field: str) -> None:
    class HostileFloat(float):
        def __float__(self) -> float:
            raise AssertionError("untrusted __float__ must not run")

        def as_integer_ratio(self) -> tuple[int, int]:
            raise AssertionError("untrusted as_integer_ratio must not run")

        def __repr__(self) -> str:
            raise AssertionError("untrusted __repr__ must not run")

    with pytest.raises(ValueError, match=field):
        SARSAConfig(n_actions=2, **{field: HostileFloat(0.25)})


@pytest.mark.parametrize("field", ("gamma", "epsilon_start", "epsilon_end"))
def test_sarsa_config_rejects_exact_nonzero_longdouble_underflow(field: str) -> None:
    nonzero = np.nextafter(np.longdouble(0.0), np.longdouble(1.0))
    with pytest.raises(ValueError, match=rf"{field}.*exact nonzero"):
        SARSAConfig(n_actions=2, **{field: nonzero})


def test_sarsa_decay_relationship_uses_exact_host_values() -> None:
    start = np.nextafter(np.longdouble(0.5), np.longdouble(0.0))
    with pytest.raises(ValueError, match="epsilon_end must not exceed"):
        SARSAConfig(
            n_actions=2,
            epsilon_start=start,
            epsilon_end=np.longdouble(0.5),
            epsilon_decay_steps=1,
        )


def test_sarsa_config_preserves_exact_builtin_zero_compatibility() -> None:
    positive = SARSAConfig(n_actions=2, gamma=0.0, epsilon_start=0.0, epsilon_end=0.0)
    negative = SARSAConfig(n_actions=2, gamma=-0.0, epsilon_start=-0.0, epsilon_end=-0.0)
    assert not np.signbit(positive.gamma)
    assert np.signbit(negative.gamma)


def test_sarsa_config_from_config_requires_exact_compatibility_schema() -> None:
    class DictSubclass(dict):
        pass

    payload = SARSAConfig(n_actions=2).to_config()
    assert SARSAConfig.from_config(payload).to_config() == payload
    with pytest.raises(ValueError, match="actual dict"):
        SARSAConfig.from_config(DictSubclass(payload))
    with pytest.raises(ValueError, match="fields"):
        SARSAConfig.from_config({**payload, "extra": 1})


def test_sarsa_agent_rejects_hostile_lambda_before_horde_construction() -> None:
    class HostileFloat:
        def __float__(self) -> float:
            raise AssertionError("untrusted __float__ must not run")

        def __repr__(self) -> str:
            raise AssertionError("untrusted __repr__ must not run")

    with pytest.raises(ValueError, match="lamda"):
        SARSAAgent(SARSAConfig(n_actions=2), lamda=HostileFloat())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["lamda", "step_size", "sparsity", "leaky_relu_slope", "utility_decay"]
)
def test_sarsa_agent_rejects_float_subclasses_without_calling_hooks(field: str) -> None:
    class HostileFloat(float):
        def __float__(self) -> float:
            raise AssertionError("untrusted __float__ must not run")

        def as_integer_ratio(self) -> tuple[int, int]:
            raise AssertionError("untrusted as_integer_ratio must not run")

        def __repr__(self) -> str:
            raise AssertionError("untrusted __repr__ must not run")

    with pytest.raises(ValueError, match=field):
        SARSAAgent(SARSAConfig(n_actions=2), **{field: HostileFloat(0.25)})


@pytest.mark.parametrize(
    "field", ("lamda", "step_size", "sparsity", "leaky_relu_slope", "utility_decay")
)
def test_sarsa_agent_rejects_exact_nonzero_longdouble_underflow(field: str) -> None:
    nonzero = np.nextafter(np.longdouble(0.0), np.longdouble(1.0))
    with pytest.raises(ValueError, match=rf"{field}.*exact nonzero"):
        SARSAAgent(
            SARSAConfig(n_actions=2),
            hidden_sizes=(),
            **{field: nonzero},
        )


def test_sarsa_agent_roundtrip_and_exact_schema() -> None:
    class DictSubclass(dict):
        pass

    agent = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=())
    payload = agent.to_config()
    restored = SARSAAgent.from_config(payload)
    assert restored.to_config() == payload
    with pytest.raises(ValueError, match="fields"):
        SARSAAgent.from_config({**payload, "extra": None})
    with pytest.raises(ValueError, match="actual dict"):
        SARSAAgent.from_config(DictSubclass(payload))


def test_sarsa_agent_from_config_rejects_prediction_demon_dict_subclasses() -> None:
    class DictSubclass(dict):
        pass

    demon = GVFSpec(
        name="prediction",
        demon_type=DemonType.PREDICTION,
        gamma=0.0,
        lamda=0.0,
        cumulant_index=0,
    )
    payload = SARSAAgent(
        SARSAConfig(n_actions=2), hidden_sizes=(), prediction_demons=[demon]
    ).to_config()
    payload["prediction_demons"][0] = DictSubclass(payload["prediction_demons"][0])
    with pytest.raises(ValueError, match="prediction_demons entries"):
        SARSAAgent.from_config(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("n_actions", np.int64(2)),
        ("epsilon_decay_steps", np.int32(0)),
        ("gamma", np.float32(0.5)),
        ("epsilon_start", 0),
    ),
)
def test_sarsa_config_loader_requires_exact_json_scalars(field: str, value: object) -> None:
    payload = SARSAConfig(n_actions=2).to_config()
    payload[field] = value
    with pytest.raises(ValueError, match="exact JSON scalar types"):
        SARSAConfig.from_config(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("hidden_sizes", [np.int64(2)]),
        ("sparsity", np.float32(0.5)),
        ("leaky_relu_slope", 0),
        ("lamda", np.float64(0.5)),
        ("utility_decay", 1),
        ("use_layer_norm", np.bool_(True)),
    ),
)
def test_sarsa_agent_loader_requires_exact_json_scalars(field: str, value: object) -> None:
    payload = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=()).to_config()
    payload[field] = value
    with pytest.raises(ValueError, match="serialized"):
        SARSAAgent.from_config(payload)


def test_sarsa_init_rejects_aggregate_state_overflow_before_jax_allocation() -> None:
    agent = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=())
    # Linear two-head state uses (5 * feature_dim + 12) direct scalars.
    first_overflow = ((2**31 - 1) // 4 - 12) // 5 + 1
    with pytest.raises(ValueError, match="aggregate_direct_state_bytes"):
        agent.init(feature_dim=first_overflow, key=jr.key(0))


def test_sarsa_constructor_rejects_impossible_hidden_state_before_demon_list() -> None:
    with pytest.raises(ValueError, match="aggregate_direct_state"):
        SARSAAgent(
            SARSAConfig(n_actions=1),
            hidden_sizes=(2**31 - 1,),
        )


def test_sarsa_step_count_saturates_without_wrapping_under_jit() -> None:
    agent = _make_agent(n_actions=2, hidden_sizes=(), epsilon_start=0.0)
    state = agent.init(feature_dim=2, key=jr.key(0)).replace(
        last_action=jnp.array(0, dtype=jnp.int32),
        last_observation=jnp.ones(2, dtype=jnp.float32),
        step_count=jnp.array(2**31 - 1, dtype=jnp.int32),
    )
    result = agent.update(
        state,
        reward=jnp.array(0.0, dtype=jnp.float32),
        observation=jnp.ones(2, dtype=jnp.float32),
        terminated=jnp.array(0.0, dtype=jnp.float32),
        next_action=jnp.array(0, dtype=jnp.int32),
    )
    assert int(result.state.step_count) == 2**31 - 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_size", float("inf")),
        ("sparsity", 1.1),
        ("leaky_relu_slope", -0.1),
        ("utility_decay", 1.0),
        ("use_layer_norm", np.bool_(True)),
    ],
)
def test_sarsa_agent_validates_all_direct_scalar_fields(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        SARSAAgent(SARSAConfig(n_actions=2), **{field: value})


def test_sarsa_agent_rejects_spoofed_static_types_and_prediction_entries() -> None:
    class TraceModeSubclass(str):
        pass

    with pytest.raises(ValueError, match="trace_mode"):
        SARSAAgent(
            SARSAConfig(n_actions=2),
            trace_mode=TraceModeSubclass("accumulating"),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="GVFSpec"):
        SARSAAgent(SARSAConfig(n_actions=2), prediction_demons=[object()])  # type: ignore[list-item]


def test_sarsa_serialized_discriminators_reject_string_subclasses() -> None:
    class StringSubclass(str):
        pass

    payload = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=()).to_config()
    payload["type"] = StringSubclass("SARSAAgent")
    with pytest.raises(ValueError, match="type"):
        SARSAAgent.from_config(payload)
    payload = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=()).to_config()
    payload["state_schema"] = StringSubclass(payload["state_schema"])
    with pytest.raises(ValueError, match="state schema"):
        SARSAAgent.from_config(payload)
    payload = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=()).to_config()
    payload["trace_mode"] = StringSubclass(payload["trace_mode"])
    with pytest.raises(ValueError, match="trace_mode"):
        SARSAAgent.from_config(payload)


def _assert_sarsa_state_exact_ignoring_host_time(actual, expected) -> None:
    actual_learner = actual.learner_state.replace(birth_timestamp=0.0, uptime_s=0.0)
    expected_learner = expected.learner_state.replace(birth_timestamp=0.0, uptime_s=0.0)
    chex.assert_trees_all_equal(
        actual.replace(learner_state=actual_learner),
        expected.replace(learner_state=expected_learner),
    )


@pytest.mark.parametrize("next_action", [-1, 2])
def test_sarsa_invalid_next_action_is_an_exact_noop(next_action: int) -> None:
    agent = _make_agent(n_actions=2, hidden_sizes=(), epsilon_start=0.0)
    observation = jnp.ones(3, dtype=jnp.float32)
    action, key = agent.select_action(agent.init(3, jr.key(0)), observation)
    state = agent.init(3, jr.key(1)).replace(
        last_action=action,
        last_observation=observation,
        rng_key=key,
    )
    result = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(next_action, dtype=jnp.int32),
    )
    _assert_sarsa_state_exact_ignoring_host_time(result.state, state)
    assert int(result.action) == -1
    assert float(result.td_error) == 0.0
    assert bool(jnp.all(result.q_values == 0.0))


@pytest.mark.parametrize(
    ("reward", "terminated"),
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (1.0, float("nan")),
        (1.0, 0.5),
    ],
)
def test_sarsa_invalid_dynamic_transition_is_an_exact_noop(reward, terminated) -> None:
    agent = _make_agent(n_actions=2, hidden_sizes=(), epsilon_start=0.0)
    observation = jnp.ones(3, dtype=jnp.float32)
    state = agent.init(3, jr.key(0)).replace(
        last_action=jnp.asarray(0, dtype=jnp.int32),
        last_observation=observation,
    )
    result = agent.update(
        state,
        jnp.asarray(reward, dtype=jnp.float32),
        observation,
        jnp.asarray(terminated, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    _assert_sarsa_state_exact_ignoring_host_time(result.state, state)
    assert int(result.action) == -1


def test_sarsa_public_array_shapes_and_dtypes_fail_before_transaction() -> None:
    agent = _make_agent(n_actions=2, hidden_sizes=(), epsilon_start=0.0)
    state = agent.init(3, jr.key(0))
    with pytest.raises(ValueError, match="observation must have shape"):
        agent.select_action(state, jnp.ones(4, dtype=jnp.float32))
    with pytest.raises(TypeError, match="observation must have dtype float32"):
        agent.select_action(state, jnp.ones(3, dtype=jnp.int32))
    with pytest.raises(TypeError, match="next_action must have dtype int32"):
        agent.update(
            state,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.ones(3, dtype=jnp.float32),
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=jnp.float32),
        )


def test_sarsa_select_action_rejects_nonfinite_without_consuming_rng() -> None:
    agent = _make_agent(n_actions=2, hidden_sizes=(), epsilon_start=0.0)
    state = agent.init(3, jr.key(0))
    action, key = agent.select_action(
        state,
        jnp.asarray([0.0, float("nan"), 1.0], dtype=jnp.float32),
    )
    assert int(action) == -1
    chex.assert_trees_all_equal(key, state.rng_key)
