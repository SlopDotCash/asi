"""Tests for the closed-loop micro-MDPs (actions affect observations)."""

from fractions import Fraction
from numbers import Real

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams import (
    LEFT_ACTION,
    PHASE_A,
    PHASE_B,
    RIGHT_ACTION,
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

_INT32_MAX = 2**31 - 1
_INVALID_PHASE_LENGTHS = (0, -1, False, True, 1.5, None, 2**31, 10**100)


class _SpoofedReward:
    """Non-real object whose ``__class__`` property impersonates ``float``."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return float

    def as_integer_ratio(self) -> tuple[int, int]:
        return (1, 2)


class _ExplodingRewardFloat(float):
    """Float subclass whose untrusted ratio hook must never execute."""

    def as_integer_ratio(self) -> tuple[int, int]:
        raise RuntimeError("untrusted reward ratio hook executed")


def _rollout_two_state(
    env: SwitchingTwoStateMDP,
    policy: tuple[int, int],
    start_state: int,
    num_steps: int,
) -> jnp.ndarray:
    """Roll out a deterministic stationary policy with ``jax.lax.scan``."""
    policy_array = jnp.asarray(policy, dtype=jnp.int32)

    def scan_fn(carry, step_key):
        state = carry
        action = policy_array[state.state_index]
        _obs, reward, new_state = env.step(state, action, step_key)
        return new_state, reward

    initial = env.init(jr.key(0)).replace(  # type: ignore[attr-defined]
        state_index=jnp.array(start_state, dtype=jnp.int32)
    )
    _final, rewards = jax.lax.scan(scan_fn, initial, jr.split(jr.key(1), num_steps))
    return rewards


# =============================================================================
# Switching two-state MDP: dynamics and rewards
# =============================================================================


class TestSwitchingTwoStateDynamics:
    """Dynamics, observations, and reward correctness."""

    def test_init_and_observe(self):
        """Initial state is a valid one-hot observation of a latent state."""
        env = SwitchingTwoStateMDP()
        state = env.init(jr.key(42))

        assert env.n_states == 2
        assert env.n_actions == 2
        assert env.feature_dim == 2
        assert int(state.step_count) == 0
        assert int(state.state_index) in (0, 1)

        obs = env.observe(state)
        chex.assert_shape(obs, (2,))
        assert obs.dtype == jnp.float32
        assert float(obs.sum()) == 1.0
        assert float(obs[int(state.state_index)]) == 1.0

    def test_actions_determine_next_observation(self):
        """The action chosen now is exactly the latent state observed next."""
        env = SwitchingTwoStateMDP()
        base = env.init(jr.key(0))
        for start in range(env.n_states):
            state = base.replace(  # type: ignore[attr-defined]
                state_index=jnp.array(start, dtype=jnp.int32)
            )
            for action in range(env.n_actions):
                obs, _reward, new_state = env.step(
                    state, jnp.array(action), jr.key(action)
                )
                assert int(new_state.state_index) == action
                assert int(new_state.step_count) == int(state.step_count) + 1
                expected = jax.nn.one_hot(action, 2, dtype=jnp.float32)
                chex.assert_trees_all_close(obs, expected)

    def test_rewards_follow_phase_a_payoffs(self):
        """At step 0 the reward for (state, action) is the phase-A payoff."""
        config = SwitchingTwoStateConfig(phase_length=100)
        env = SwitchingTwoStateMDP(config)
        base = env.init(jr.key(0))
        for start in range(2):
            state = base.replace(  # type: ignore[attr-defined]
                state_index=jnp.array(start, dtype=jnp.int32)
            )
            for action in range(2):
                _obs, reward, _new = env.step(state, jnp.array(action), jr.key(0))
                assert float(reward) == config.payoffs_a[start][action]

    def test_phase_switch_schedule(self):
        """The phase follows A -> B -> A with period ``phase_length``."""
        env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=5))
        state = env.init(jr.key(3))
        phases = []
        for step in range(15):
            phases.append(int(env.phase_id(state)))
            _obs, _reward, state = env.step(state, jnp.array(0), jr.key(step))
        assert phases == [PHASE_A] * 5 + [PHASE_B] * 5 + [PHASE_A] * 5

    def test_reward_structure_actually_switches(self):
        """The same (state, action) pair pays differently in phase A and B."""
        env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=3))
        state = env.init(jr.key(0)).replace(  # type: ignore[attr-defined]
            state_index=jnp.array(0, dtype=jnp.int32)
        )
        # In state 0, action 1 pays 1.0 under phase A and 0.0 under phase B.
        _obs, reward_a, _new = env.step(state, jnp.array(1), jr.key(0))
        in_phase_b = state.replace(  # type: ignore[attr-defined]
            step_count=jnp.array(3, dtype=jnp.int32)
        )
        _obs, reward_b, _new = env.step(in_phase_b, jnp.array(1), jr.key(0))
        assert float(reward_a) == 1.0
        assert float(reward_b) == 0.0

    @pytest.mark.parametrize("phase_length", _INVALID_PHASE_LENGTHS)
    def test_invalid_phase_length_raises(self, phase_length):
        """Schedule divisors must be built-in positive JAX-int32 integers."""
        with pytest.raises(ValueError, match=r"phase_length must be"):
            SwitchingTwoStateMDP(
                SwitchingTwoStateConfig(phase_length=phase_length)  # type: ignore[arg-type]
            )

    def test_phase_length_class_spoofed_int_raises(self):
        """A ``__class__`` override that fakes ``int`` must still be rejected.

        ``isinstance(value, int)`` consults the overridable ``__class__``
        property, so a non-int object could previously spoof the isinstance
        check and reach ``int(config.phase_length)`` unvalidated, raising an
        undocumented ``TypeError`` from deep inside JAX construction instead
        of the clean ``ValueError`` this constructor promises.
        """

        class _SpoofedInt:
            @property
            def __class__(self) -> type:  # noqa: A003 - deliberate spoof target
                return int

            def __lt__(self, other: object) -> bool:
                return False

            def __gt__(self, other: object) -> bool:
                return False

        with pytest.raises(ValueError, match=r"phase_length must be a built-in integer"):
            SwitchingTwoStateMDP(
                SwitchingTwoStateConfig(phase_length=_SpoofedInt())  # type: ignore[arg-type]
            )

    def test_int32_max_phase_length_runs_first_eager_and_jit_query(self):
        """The largest JAX-int32 phase divisor is accepted without overflow."""
        env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=_INT32_MAX))
        state = env.init(jr.key(0))
        assert int(env.phase_id(state)) == PHASE_A
        assert int(jax.jit(env.phase_id)(state)) == PHASE_A

    def test_invalid_payoff_shape_raises(self):
        """Payoff matrices must preserve the fixed state/action shape."""
        with pytest.raises(ValueError, match="2x2"):
            SwitchingTwoStateMDP(
                SwitchingTwoStateConfig(payoffs_a=((0.0, 1.0, 2.0),) * 2)  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "payoffs_a",
        [
            ((float("nan"), 0.0), (0.0, 1.0)),
            ((float("inf"), 0.0), (0.0, 1.0)),
            ((-1.0, 0.0), (0.0, float("-inf"))),
        ],
    )
    def test_non_finite_payoffs_raise(self, payoffs_a):
        """Payoff matrices must contain only finite values."""
        with pytest.raises(ValueError, match="finite"):
            SwitchingTwoStateMDP(SwitchingTwoStateConfig(payoffs_a=payoffs_a))

    def test_non_finite_payoffs_b_raise(self):
        """payoffs_b is validated like payoffs_a."""
        with pytest.raises(ValueError, match="finite"):
            SwitchingTwoStateMDP(
                SwitchingTwoStateConfig(
                    payoffs_b=((0.0, float("nan")), (1.0, 0.0))  # type: ignore[arg-type]
                )
            )


# =============================================================================
# Switching two-state MDP: analytic helpers
# =============================================================================


class TestSwitchingAnalyticHelpers:
    """Closed-form optimal and uniform-random average rewards."""

    def test_default_payoffs_optimum_and_baseline(self):
        """Default phases both have optimum 1.0 and random baseline 0.5."""
        env = SwitchingTwoStateMDP()
        assert env.optimal_average_reward(PHASE_A) == 1.0
        assert env.optimal_average_reward(PHASE_B) == 1.0
        assert env.uniform_random_average_reward(PHASE_A) == 0.5
        assert env.uniform_random_average_reward(PHASE_B) == 0.5

    def test_custom_payoffs_pick_best_cycle(self):
        """The optimum is the best of the two self-loops and the toggle cycle."""
        toggle_best = SwitchingTwoStateMDP(
            SwitchingTwoStateConfig(payoffs_a=((0.2, 0.9), (0.4, 0.1)))
        )
        assert toggle_best.optimal_average_reward(PHASE_A) == pytest.approx(0.65)

        stay_best = SwitchingTwoStateMDP(
            SwitchingTwoStateConfig(payoffs_a=((0.7, 0.1), (0.0, 0.3)))
        )
        assert stay_best.optimal_average_reward(PHASE_A) == pytest.approx(0.7)

    def test_optimal_matches_brute_force_rollouts(self):
        """The closed form equals the best empirical deterministic policy."""
        env = SwitchingTwoStateMDP(
            SwitchingTwoStateConfig(
                phase_length=10_000, payoffs_a=((0.2, 0.9), (0.4, 0.1))
            )
        )
        num_steps = 200
        empirical = [
            float(_rollout_two_state(env, policy, start, num_steps).mean())
            for policy in ((0, 0), (0, 1), (1, 0), (1, 1))
            for start in (0, 1)
        ]
        # Transients contribute at most 1/num_steps to any average.
        assert max(empirical) == pytest.approx(
            env.optimal_average_reward(PHASE_A), abs=1.0 / num_steps
        )

    def test_uniform_baseline_matches_simulation(self):
        """A uniform-random rollout attains the analytic baseline."""
        env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=10_000))

        def scan_fn(carry, step_key):
            state = carry
            action_key, step_key = jr.split(step_key)
            action = jr.randint(action_key, (), 0, 2)
            _obs, reward, new_state = env.step(state, action, step_key)
            return new_state, reward

        _final, rewards = jax.lax.scan(
            scan_fn, env.init(jr.key(0)), jr.split(jr.key(1), 4000)
        )
        assert float(rewards.mean()) == pytest.approx(
            env.uniform_random_average_reward(PHASE_A), abs=0.05
        )

    def test_invalid_phase_raises(self):
        """Phases other than PHASE_A/PHASE_B are rejected."""
        env = SwitchingTwoStateMDP()
        with pytest.raises(ValueError, match="phase"):
            env.optimal_average_reward(2)


# =============================================================================
# Switching two-state MDP: scan compatibility
# =============================================================================


class TestSwitchingScanRollout:
    """Full jit + lax.scan rollouts under a fixed policy."""

    def test_jitted_scan_rollout_crosses_phases(self):
        """A jitted scan rollout of the toggle policy spans a phase switch."""
        phase_length = 250
        env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=phase_length))

        @jax.jit
        def rollout(key):
            init_key, scan_key = jr.split(key)

            def scan_fn(carry, step_key):
                state = carry
                action = 1 - state.state_index  # toggle policy
                obs, reward, new_state = env.step(state, action, step_key)
                return new_state, (obs, reward)

            final, (observations, rewards) = jax.lax.scan(
                scan_fn, env.init(init_key), jr.split(scan_key, 2 * phase_length)
            )
            return final, observations, rewards

        final, observations, rewards = rollout(jr.key(7))

        chex.assert_shape(observations, (2 * phase_length, 2))
        chex.assert_shape(rewards, (2 * phase_length,))
        chex.assert_tree_all_finite((observations, rewards))
        assert int(final.step_count) == 2 * phase_length
        # Observations stay one-hot along the whole rollout.
        np.testing.assert_allclose(np.asarray(observations.sum(axis=1)), 1.0)
        # Toggling is optimal in phase A (average 1.0) and pessimal in
        # phase B (average 0.0) under the default payoffs.
        assert float(rewards[:phase_length].mean()) == pytest.approx(1.0)
        assert float(rewards[phase_length:].mean()) == pytest.approx(0.0)


# =============================================================================
# RiverSwim-style stochastic chain
# =============================================================================


class TestRiverSwim:
    """Dynamics, rewards, and analytic helpers of the stochastic variant."""

    @pytest.mark.parametrize("reward_left", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_reward_left_raises(self, reward_left):
        """reward_left must be finite."""
        with pytest.raises(ValueError, match="reward_left must be finite"):
            RiverSwimMDP(RiverSwimConfig(reward_left=reward_left))

    @pytest.mark.parametrize("reward_right", [float("nan"), float("inf")])
    def test_non_finite_reward_right_raises(self, reward_right):
        """reward_right must be finite."""
        with pytest.raises(ValueError, match="reward_right must be finite"):
            RiverSwimMDP(RiverSwimConfig(reward_right=reward_right))

    @pytest.mark.parametrize("field", ["reward_left", "reward_right"])
    @pytest.mark.parametrize(
        "value",
        [
            True,
            np.bool_(False),
            "0.5",
            object(),
            _SpoofedReward(),
            _ExplodingRewardFloat(0.5),
            1.0e100,
            -1.0e100,
        ],
        ids=(
            "bool",
            "numpy-bool",
            "string",
            "object",
            "class-spoof",
            "exploding-ratio",
            "positive-overflow",
            "negative-overflow",
        ),
    )
    def test_rewards_reject_untrusted_or_non_float32_values(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            RiverSwimMDP(RiverSwimConfig(**{field: value}))  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["reward_left", "reward_right"])
    def test_rewards_are_canonicalized_directly_to_json_safe_float32(
        self,
        field: str,
    ) -> None:
        midpoint_plus = Fraction(1) + Fraction(1, 1 << 24) + Fraction(1, 1 << 60)
        expected = float(np.nextafter(np.float32(1.0), np.float32(2.0)))

        env = RiverSwimMDP(RiverSwimConfig(**{field: midpoint_plus}))  # type: ignore[arg-type]
        stored = getattr(env.config, field)

        assert type(stored) is float
        assert stored == expected
        assert np.isfinite(np.asarray(env.reward_tensor)).all()

    @pytest.mark.parametrize("initial_state", [1.5, True, 2.0])
    def test_non_integer_initial_state_raises(self, initial_state):
        """initial_state must be a canonical integer in range."""
        with pytest.raises(ValueError, match="initial_state must be a built-in integer"):
            RiverSwimMDP(RiverSwimConfig(initial_state=initial_state))  # type: ignore[arg-type]

    def test_initial_state_class_spoofed_int_raises(self):
        """A ``__class__`` override that fakes ``int`` must still be rejected.

        Previously ``isinstance(config.initial_state, Integral)`` consulted
        the overridable ``__class__`` property, so this object would pass
        validation and reach ``jnp.array(self._config.initial_state, ...)``
        inside :meth:`RiverSwimMDP.init` unvalidated.
        """

        class _SpoofedInt:
            @property
            def __class__(self) -> type:  # noqa: A003 - deliberate spoof target
                return int

            def __ge__(self, other: object) -> bool:
                return True

            def __lt__(self, other: object) -> bool:
                return True

        with pytest.raises(ValueError, match="initial_state must be a built-in integer"):
            RiverSwimMDP(RiverSwimConfig(initial_state=_SpoofedInt()))  # type: ignore[arg-type]

    def test_non_integer_n_states_raises(self):
        """n_states must be a canonical built-in integer, not merely numeric.

        Previously ``n_states`` had no type check at all: a float such as
        ``6.5`` passed the ``config.n_states < 2`` comparison and then raised
        an undocumented ``TypeError`` from ``range(n)`` deep inside
        ``_build_transitions`` instead of a clean ``ValueError``.
        """
        with pytest.raises(ValueError, match="n_states must be a built-in integer"):
            RiverSwimMDP(RiverSwimConfig(n_states=6.5))  # type: ignore[arg-type]

    def test_transition_tensor_structure(self):
        """Kernels are row-stochastic with drift folded at the boundaries."""
        config = RiverSwimConfig(n_states=4, p_right_up=0.3, p_right_down=0.1)
        env = RiverSwimMDP(config)
        transitions = env.transition_tensor

        chex.assert_shape(transitions, (2, 4, 4))
        np.testing.assert_allclose(transitions.sum(axis=2), 1.0, atol=1e-6)
        # LEFT is deterministic one step left, saturating at state 0.
        np.testing.assert_allclose(transitions[LEFT_ACTION, 0], [1, 0, 0, 0])
        np.testing.assert_allclose(transitions[LEFT_ACTION, 2], [0, 1, 0, 0])
        # RIGHT from a middle state: down / stay / up.
        np.testing.assert_allclose(
            transitions[RIGHT_ACTION, 1], [0.1, 0.6, 0.3, 0.0], atol=1e-6
        )
        # Boundary folding: no leftward move at 0, no rightward move at the top.
        np.testing.assert_allclose(
            transitions[RIGHT_ACTION, 0], [0.7, 0.3, 0.0, 0.0], atol=1e-6
        )
        np.testing.assert_allclose(
            transitions[RIGHT_ACTION, 3], [0.0, 0.0, 0.1, 0.9], atol=1e-6
        )

    def test_rewards_only_at_chain_ends(self):
        """Reward is reward_left at (0, LEFT), reward_right at (top, RIGHT)."""
        env = RiverSwimMDP(RiverSwimConfig(n_states=4))
        rewards = env.reward_tensor
        expected = np.zeros((4, 2), dtype=np.float32)
        expected[0, LEFT_ACTION] = env.config.reward_left
        expected[3, RIGHT_ACTION] = env.config.reward_right
        np.testing.assert_allclose(rewards, expected)

    def test_step_matches_kernel_empirically(self):
        """Vmapped single steps from a middle state match the RIGHT kernel row."""
        env = RiverSwimMDP(RiverSwimConfig(n_states=5))
        state = env.init(jr.key(0)).replace(  # type: ignore[attr-defined]
            state_index=jnp.array(2, dtype=jnp.int32)
        )

        def one_step(key):
            obs, reward, new_state = env.step(state, jnp.array(RIGHT_ACTION), key)
            return new_state.state_index, reward

        next_states, rewards = jax.vmap(one_step)(jr.split(jr.key(1), 4000))

        frequencies = np.bincount(np.asarray(next_states), minlength=5) / 4000.0
        np.testing.assert_allclose(
            frequencies, env.transition_tensor[RIGHT_ACTION, 2], atol=0.03
        )
        # Middle states never pay.
        assert float(jnp.abs(rewards).max()) == 0.0

    def test_left_action_is_deterministic(self):
        """LEFT always moves exactly one state left and returns its one-hot."""
        env = RiverSwimMDP(RiverSwimConfig(n_states=5))
        state = env.init(jr.key(0)).replace(  # type: ignore[attr-defined]
            state_index=jnp.array(3, dtype=jnp.int32)
        )
        for seed in range(5):
            obs, _reward, new_state = env.step(
                state, jnp.array(LEFT_ACTION), jr.key(seed)
            )
            assert int(new_state.state_index) == 2
            chex.assert_trees_all_close(obs, jax.nn.one_hot(2, 5, dtype=jnp.float32))

    def test_optimal_policy_is_always_right(self):
        """With default parameters the gain-optimal policy swims right."""
        env = RiverSwimMDP(RiverSwimConfig(n_states=4))
        assert env.optimal_policy() == (RIGHT_ACTION,) * 4

        optimal = env.optimal_average_reward()
        always_left = env.policy_average_reward([LEFT_ACTION] * 4)
        uniform = env.uniform_random_average_reward()
        assert optimal == pytest.approx(
            env.policy_average_reward([RIGHT_ACTION] * 4)
        )
        assert always_left == pytest.approx(env.config.reward_left)
        assert uniform < optimal
        assert always_left < optimal

    def test_policy_gain_matches_scan_simulation(self):
        """A long scan rollout of always-right attains its analytic gain."""
        env = RiverSwimMDP(RiverSwimConfig(n_states=4))

        def scan_fn(carry, step_key):
            state = carry
            _obs, reward, new_state = env.step(
                state, jnp.array(RIGHT_ACTION), step_key
            )
            return new_state, reward

        _final, rewards = jax.lax.scan(
            scan_fn, env.init(jr.key(0)), jr.split(jr.key(1), 50_000)
        )
        # Discard burn-in so the empirical average reflects the stationary chain.
        empirical = float(rewards[5_000:].mean())
        assert empirical == pytest.approx(
            env.policy_average_reward([RIGHT_ACTION] * 4), abs=0.02
        )

    def test_invalid_config_raises(self):
        """Chain length, drift, and start-state validation."""
        with pytest.raises(ValueError, match="n_states"):
            RiverSwimMDP(RiverSwimConfig(n_states=1))
        with pytest.raises(ValueError, match="p_right_up must be finite"):
            RiverSwimMDP(RiverSwimConfig(p_right_up=float("nan")))
        with pytest.raises(ValueError, match="p_right_down must be finite"):
            RiverSwimMDP(RiverSwimConfig(p_right_down=float("nan")))
        with pytest.raises(ValueError, match="p_right_down"):
            RiverSwimMDP(RiverSwimConfig(p_right_down=0.0))
        with pytest.raises(ValueError, match="must not exceed 1"):
            RiverSwimMDP(RiverSwimConfig(p_right_up=0.7, p_right_down=0.4))
        with pytest.raises(ValueError, match="initial_state"):
            RiverSwimMDP(RiverSwimConfig(n_states=3, initial_state=3))
        with pytest.raises(ValueError, match="policy"):
            RiverSwimMDP(RiverSwimConfig(n_states=3)).policy_average_reward([0, 1])

    @pytest.mark.parametrize(
        "value",
        [
            True,
            False,
            "0.2",
            None,
            10**400,
            1.0e-50,
            np.nextafter(np.longdouble(0.0), np.longdouble(1.0)),
            jnp.asarray(0.2),
            jnp.asarray([0.2]),
        ],
    )
    @pytest.mark.parametrize("field", ["p_right_up", "p_right_down"])
    def test_transition_probabilities_require_positive_float32_reals(
        self,
        field,
        value,
    ):
        kwargs = {field: value}
        with pytest.raises(ValueError, match=field):
            RiverSwimMDP(RiverSwimConfig(**kwargs))

    @pytest.mark.parametrize("field", ["p_right_up", "p_right_down"])
    def test_transition_probabilities_reject_class_spoofed_reals(self, field):
        """``__class__``-spoofed non-``Real`` objects must not defeat validation."""

        class _SpoofedFloat:
            """Mimics ``float`` via ``__class__`` to defeat ``isinstance``."""

            @property
            def __class__(self) -> type:  # type: ignore[override]
                return float

            def __float__(self) -> float:
                return 0.3

            def as_integer_ratio(self) -> tuple[int, int]:
                return (3, 10)

        assert isinstance(_SpoofedFloat(), Real)
        assert not issubclass(type(_SpoofedFloat()), Real)

        kwargs = {field: _SpoofedFloat()}
        with pytest.raises(ValueError, match=field):
            RiverSwimMDP(RiverSwimConfig(**kwargs))  # type: ignore[arg-type]

    def test_transition_probabilities_preserve_real_scalars_and_normalize_runtime(self):
        env = RiverSwimMDP(
            RiverSwimConfig(
                p_right_up=Fraction(1, 5),
                p_right_down=np.float64(0.1),
            )
        )

        assert type(env.config.p_right_up) is float
        assert type(env.config.p_right_down) is float
        assert env.config.p_right_up == float(np.float32(0.2))
        assert env.config.p_right_down == float(np.float32(0.1))
        np.testing.assert_allclose(env.transition_tensor.sum(axis=2), 1.0, atol=1e-7)

    def test_float32_probability_sum_cannot_create_invalid_stay_mass(self):
        with pytest.raises(ValueError, match="must not exceed 1"):
            RiverSwimMDP(
                RiverSwimConfig(
                    p_right_up=0.6,
                    p_right_down=0.4,
                )
            )

    @pytest.mark.parametrize(
        ("p_right_up", "p_right_down"),
        [
            (
                np.nextafter(np.longdouble(0.5), np.longdouble(1.0)),
                np.longdouble(0.5),
            ),
            (Fraction((2**100) + 1, 2**101), Fraction(1, 2)),
        ],
    )
    def test_transition_probability_sum_is_checked_before_narrowing(
        self,
        p_right_up,
        p_right_down,
    ):
        with pytest.raises(ValueError, match="must not exceed 1"):
            RiverSwimMDP(
                RiverSwimConfig(
                    p_right_up=p_right_up,
                    p_right_down=p_right_down,
                )
            )
