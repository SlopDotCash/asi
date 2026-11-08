"""Semi-MDP degeneracy tests for the STOMP agent (Alberta Plan Step 10).

The semi-MDP formulation must degenerate exactly to primitive differential
Q-learning when every option lasts exactly one primitive step and the
intra-option discount is 1.  Two grounding defects break this property:

1. The base extended-Q update on option termination must use the environment
   reward accumulated across the option (task-reward units), never the
   subtask pseudo-reward.
2. The intra-option learner must only update while an option is actually
   executing; idle steps must not pollute option 0 (the clamped index).
"""

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from alberta_framework.core.options import (
    STOMPAgent,
    STOMPConfig,
    SubtaskSpec,
    _select_action_epsilon_greedy_from_q,
)

OBS_DIM = 4
N_PRIMITIVE = 2
UNREACHABLE_THRESHOLD = 1.0e6  # observations are in [0, 1); goal never fires


def _degenerate_config(n_options: int = 2, **overrides: object) -> STOMPConfig:
    """STOMP config whose options degenerate to one-step macro-actions."""
    specs = tuple(
        SubtaskSpec(
            feature_index=i,
            threshold=UNREACHABLE_THRESHOLD,
            max_option_steps=1,
        )
        for i in range(n_options)
    )
    kwargs: dict[str, object] = dict(
        subtask_specs=specs,
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        option_gamma=1.0,
        epsilon_base=0.3,
    )
    kwargs.update(overrides)
    return STOMPConfig(**kwargs)  # type: ignore[arg-type]


def _transition_stream(key, num_steps: int):
    """Pre-generated (rewards, observations) with rewards != pseudo-rewards."""
    r_key, o_key = jr.split(key)
    rewards = 1.5 * jr.normal(r_key, (num_steps,), dtype=jnp.float32) + 0.5
    observations = jr.uniform(o_key, (num_steps, OBS_DIM), dtype=jnp.float32)
    return rewards, observations


class TestSemiMDPDegeneracy:
    """One-step options with gamma_o=1 must equal primitive Q-learning."""

    @pytest.mark.parametrize("base_trace_decay", [0.0, 0.5])
    @pytest.mark.parametrize("transition_discounts", [(1.0,), (1.0, 0.5, 0.0, 0.75)])
    def test_base_q_trajectory_matches_primitive_learner(
        self, base_trace_decay, transition_discounts,
    ):
        """Trajectory of base-Q updates equals a primitive differential learner.

        With ``max_option_steps=1``, unreachable thresholds, and
        ``option_gamma=1.0`` every extended action (primitive or option) spans
        exactly one environment transition, so the STOMP base agent must be
        indistinguishable from differential Q-learning over the extended
        action set: same TD errors, same average-reward trace, same action
        sequence, same final weights.
        """
        num_steps = 80
        config = _degenerate_config(base_trace_decay=base_trace_decay)
        agent = STOMPAgent(config)
        learner = agent.base_learner
        n_total = config.n_total_actions
        beta = jnp.asarray(config.base_avg_reward_step_size, dtype=jnp.float32)

        rewards, observations = _transition_stream(jr.key(7), num_steps + 1)
        obs0 = observations[0]

        state = agent.start(agent.init(jr.key(3)), obs0)

        # Reference primitive differential Q-learner, seeded from the exact
        # same initial conditions and mirroring the agent's RNG consumption.
        ref_learner_state = state.base_learner_state
        ref_avg_reward = state.base_average_reward
        ref_last_obs = state.base_last_obs
        ref_last_action = state.base_last_action
        ref_key = state.rng_key
        ref_previous_discount = jnp.asarray(1.0, dtype=jnp.float32)

        stomp_tds, stomp_avgs, stomp_actions, stomp_terms = [], [], [], []
        ref_tds, ref_avgs, ref_actions = [], [], []

        for t in range(1, num_steps + 1):
            reward, obs = rewards[t], observations[t]
            discount = jnp.asarray(
                transition_discounts[(t - 1) % len(transition_discounts)], dtype=jnp.float32,
            )

            result = agent.update(state, reward, obs, discount)
            state = result.state
            stomp_tds.append(float(result.td_error))
            stomp_avgs.append(float(result.average_reward))
            stomp_actions.append(int(state.base_last_action))
            stomp_terms.append(bool(result.option_terminated))

            # Primitive differential Q-learning over the extended action set.
            next_q = learner.predict(ref_learner_state, obs)
            td_target = reward - ref_avg_reward + discount * jnp.max(next_q)
            targets = jnp.full(n_total, jnp.nan, dtype=jnp.float32).at[
                ref_last_action
            ].set(td_target)
            # Eligibility uses the discount into the state being trained.
            # The current transition discount enters the bootstrap above.
            ref_learner_state = ref_learner_state.replace(
                head_traces=jtu.tree_map(
                    lambda trace: ref_previous_discount * trace,
                    ref_learner_state.head_traces,
                )
            )
            upd = learner.update(ref_learner_state, ref_last_obs, targets)
            td = upd.errors[ref_last_action]
            ref_avg_reward = ref_avg_reward + beta * td
            # The multihead update only advances the selected action. The
            # primitive reference clock also decays every inactive trace
            # once per transition without adding a gradient to that trace.
            active_head = int(ref_last_action)
            ref_traces = tuple(
                updated if head == active_head else (
                    base_trace_decay * previous[0], base_trace_decay * previous[1]
                )
                for head, (previous, updated) in enumerate(zip(
                    ref_learner_state.head_traces, upd.state.head_traces, strict=True,
                ))
            )
            ref_learner_state = upd.state.replace(head_traces=ref_traces)
            # Mirror the agent's per-step split (intra_key is unused here).
            ref_key, ext_key, _intra_key = jr.split(ref_key, 3)
            q_vals = learner.predict(ref_learner_state, obs)
            action, _ = _select_action_epsilon_greedy_from_q(
                q_vals, ext_key, config.epsilon_base, n_total
            )
            ref_tds.append(float(td))
            ref_avgs.append(float(ref_avg_reward))
            ref_actions.append(int(action))
            ref_last_obs = obs
            ref_last_action = action
            ref_previous_discount = discount

        # The degeneracy is only meaningful if option executions occurred.
        assert any(stomp_terms), "no option was ever executed; test is vacuous"

        assert stomp_actions == ref_actions
        np.testing.assert_allclose(stomp_tds, ref_tds, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(stomp_avgs, ref_avgs, rtol=1e-5, atol=1e-5)
        for actual, expected in zip(
            jtu.tree_leaves(state.base_learner_state),
            jtu.tree_leaves(ref_learner_state),
        ):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5
            )


class TestIdleStepsDoNotTouchOptionLearner:
    """Base-agent (idle) transitions must not update any intra-option policy."""

    def test_option_policies_unchanged_while_idle(self):
        config = _degenerate_config(n_options=1)
        agent = STOMPAgent(config)
        rewards, observations = _transition_stream(jr.key(11), 5)

        state = agent.start(agent.init(jr.key(5)), observations[0])

        for t in range(1, 5):
            # Force the pre-update state to be idle regardless of what the
            # previous step selected.
            state = state.replace(
                executing_option=jnp.array(-1, dtype=jnp.int32),
                base_last_action=state.last_primitive_action,
            )
            assert bool(agent.state_valid(state))
            before = state.option_policies
            result = agent.update(state, rewards[t], observations[t])
            assert bool(result.update_applied)
            state = result.state
            after = state.option_policies

            np.testing.assert_array_equal(
                np.asarray(after.q_weights), np.asarray(before.q_weights)
            )
            np.testing.assert_array_equal(
                np.asarray(after.traces), np.asarray(before.traces)
            )
            np.testing.assert_array_equal(
                np.asarray(after.average_rewards), np.asarray(before.average_rewards)
            )


class TestOptionTerminationGrounding:
    """Base-Q target on termination uses accumulated ENV reward, not pseudo."""

    def test_multi_step_option_accumulates_env_reward(self):
        """A 3-step option uses matched discounted reward/baseline and gamma**T."""
        gamma_o = 0.5
        config = STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=UNREACHABLE_THRESHOLD,
                    max_option_steps=3,
                ),
            ),
            observation_dim=OBS_DIM,
            n_primitive_actions=N_PRIMITIVE,
            option_gamma=gamma_o,
            epsilon_base=0.0,
        )
        agent = STOMPAgent(config)
        learner = agent.base_learner
        option_action = jnp.array(N_PRIMITIVE, dtype=jnp.int32)  # extended idx of o_0

        _, observations = _transition_stream(jr.key(13), 4)
        env_rewards = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float32)

        state = agent.start(agent.init(jr.key(17)), observations[0])
        # Force the agent into freshly-started option 0.
        state = state.replace(
            executing_option=jnp.array(0, dtype=jnp.int32),
            base_last_action=option_action,
            option_start_obs=observations[0],
            option_last_intra_action=state.last_primitive_action,
            option_cumreward=jnp.array(0.0, dtype=jnp.float32),
            option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
            option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
            option_discount=jnp.array(1.0, dtype=jnp.float32),
            option_steps=jnp.array(0, dtype=jnp.int32),
        )

        # Steps 1-2: option executing, not terminating -> no base update.
        for t in range(1, 3):
            result = agent.update(state, env_rewards[t - 1], observations[t])
            state = result.state
            assert not bool(result.option_terminated)
            assert int(state.executing_option) == 0

        # Step 3 terminates (max_option_steps=3). The discounted differential
        # target applies the same powers to reward and average-reward baseline:
        # Σ gamma^k*r_k - rbar*Σ gamma^k + gamma^3*max Q(s').
        avg_reward = state.base_average_reward
        pre_learner_state = state.base_learner_state
        q_last = learner.predict(pre_learner_state, state.option_start_obs)[option_action]
        max_next_q = jnp.max(learner.predict(pre_learner_state, observations[3]))
        discounted_env_return = sum(
            gamma_o**k * env_rewards[k] for k in range(len(env_rewards))
        )
        discounted_baseline_mass = sum(
            gamma_o**k for k in range(len(env_rewards))
        )
        expected_target = (
            discounted_env_return
            - avg_reward * discounted_baseline_mass
            + (gamma_o**3) * max_next_q
        )
        expected_td = expected_target - q_last

        result = agent.update(state, env_rewards[2], observations[3])
        assert bool(result.option_terminated)
        np.testing.assert_allclose(
            float(result.td_error), float(expected_td), rtol=1e-5, atol=1e-5
        )
