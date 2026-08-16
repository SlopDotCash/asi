"""Tests for nonlinear off-policy Horde backend."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.off_policy_horde import (
    NonlinearSharedGTDHordeLearner,
    OffPolicyHordeLearner,
    OffPolicyHordeUpdateResult,
    run_off_policy_horde_learning_loop,
)
from alberta_framework.core.optimizers import LMS, ObGDBounding
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec, TraceMode, create_horde_spec


def _spec(
    gammas: tuple[float, ...] = (0.0, 0.8),
    lamdas: tuple[float, ...] | None = None,
) -> HordeSpec:
    if lamdas is None:
        lamdas = tuple(0.0 for _ in gammas)
    demons = tuple(
        GVFSpec(
            name=f"demon_{i}",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=lamdas[i],
            cumulant_index=i,
        )  # type: ignore[call-arg]
        for i, gamma in enumerate(gammas)
    )
    return create_horde_spec(demons)


def test_init_predict_and_config_roundtrip() -> None:
    learner = OffPolicyHordeLearner(
        _spec(),
        hidden_sizes=(8,),
        optimizer=LMS(step_size=0.03),
        ratio_clip=2.0,
        trace_ratio_clip=1.0,
    )
    state = learner.init(3, jax.random.key(0))
    preds = learner.predict(state, jnp.ones(3, dtype=jnp.float32))

    chex.assert_shape(preds, (2,))
    chex.assert_tree_all_finite(preds)

    config = learner.to_config()
    restored = OffPolicyHordeLearner.from_config(config)
    assert restored.to_config() == config


def test_invalid_clips_raise() -> None:
    with pytest.raises(ValueError, match="ratio_clip"):
        OffPolicyHordeLearner(_spec(), ratio_clip=0.0)
    with pytest.raises(ValueError, match="trace_ratio_clip"):
        OffPolicyHordeLearner(_spec(), trace_ratio_clip=-1.0)
    with pytest.raises(ValueError, match="min_behavior_probability"):
        OffPolicyHordeLearner(_spec(), min_behavior_probability=0.0)


def test_update_finite_and_shapes() -> None:
    learner = OffPolicyHordeLearner(
        _spec(),
        hidden_sizes=(6,),
        optimizer=LMS(step_size=0.01),
        ratio_clip=1.5,
    )
    state = learner.init(4, jax.random.key(1))
    result = learner.update_with_ratios(
        state,
        jnp.array([0.2, -0.1, 0.4, 0.3], dtype=jnp.float32),
        jnp.array([1.0, -0.5], dtype=jnp.float32),
        jnp.array([0.0, 0.1, 0.2, -0.3], dtype=jnp.float32),
        jnp.array([2.0, 0.5], dtype=jnp.float32),
    )

    assert isinstance(result, OffPolicyHordeUpdateResult)
    chex.assert_shape(result.predictions, (2,))
    chex.assert_shape(result.next_predictions, (2,))
    chex.assert_shape(result.td_errors, (2,))
    chex.assert_shape(result.clipped_rhos, (2,))
    chex.assert_shape(result.per_demon_metrics, (2, 6))
    chex.assert_tree_all_finite(result.state.trunk_params)
    chex.assert_tree_all_finite(result.state.head_params)
    assert float(result.clipped_rhos[0]) == pytest.approx(1.5)


def test_ratio_zero_blocks_that_demon_update() -> None:
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0, 0.0)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        ratio_clip=10.0,
        sparsity=0.0,
    )
    state = learner.init(2, jax.random.key(2))
    before_w0 = state.head_params.weights[0]
    before_b0 = state.head_params.biases[0]
    before_w1 = state.head_params.weights[1]

    result = learner.update_with_ratios(
        state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([5.0, 5.0], dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.array([0.0, 2.0], dtype=jnp.float32),
    )

    chex.assert_trees_all_close(result.state.head_params.weights[0], before_w0)
    chex.assert_trees_all_close(result.state.head_params.biases[0], before_b0)
    assert float(jnp.linalg.norm(result.state.head_params.weights[1] - before_w1)) > 0.0
    assert float(result.clipped_rhos[0]) == pytest.approx(0.0)
    assert float(result.clipped_rhos[1]) == pytest.approx(2.0)


def test_infinite_cumulant_with_obgd_does_not_poison_head() -> None:
    """Inf TD error zeros the ObGD step, then error_i*step is 0*inf=NaN."""
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0, 0.0)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        bounder=ObGDBounding(kappa=2.0),
        ratio_clip=10.0,
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(2, jax.random.key(0))
    before_w0 = state.head_params.weights[0]
    before_w1 = state.head_params.weights[1]

    poisoned = learner.update_with_ratios(
        state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([jnp.inf, 1.0], dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    assert bool(jnp.all(jnp.isfinite(poisoned.state.head_params.weights[0])))
    chex.assert_trees_all_close(poisoned.state.head_params.weights[0], before_w0)
    assert bool(jnp.all(jnp.isfinite(poisoned.state.head_params.weights[1])))
    assert float(
        jnp.linalg.norm(poisoned.state.head_params.weights[1] - before_w1)
    ) > 0.0
    assert int(poisoned.state.step_count) == int(state.step_count) + 1
    assert bool(poisoned.update_applied)
    chex.assert_trees_all_equal(
        poisoned.head_updates_applied,
        jnp.array([False, True]),
    )
    assert float(poisoned.td_errors[0]) == 0.0

    recovered = learner.update_with_ratios(
        poisoned.state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([1.0, 0.5], dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(recovered.state.head_params)
    assert bool(recovered.update_applied)
    assert bool(jnp.all(recovered.head_updates_applied))


def test_zero_discount_ignores_nonfinite_next_observation_under_jit() -> None:
    """A terminal head must not evaluate an irrelevant bootstrap value."""
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0,)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(2, jax.random.key(19))
    observation = jnp.array([1.0, -0.5], dtype=jnp.float32)
    nonfinite_next = jnp.array([jnp.inf, 0.0], dtype=jnp.float32)
    assert not bool(jnp.isfinite(learner.predict(state, nonfinite_next)[0]))

    result = learner.update_with_ratios_and_discounts(
        state,
        observation,
        jnp.array([1.25], dtype=jnp.float32),
        nonfinite_next,
        jnp.ones(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.head_updates_applied, jnp.array([True]))
    chex.assert_trees_all_close(result.td_targets, jnp.array([1.25]))
    chex.assert_trees_all_close(result.next_predictions, jnp.zeros(1))
    chex.assert_tree_all_finite(result.state)
    assert int(result.state.step_count) == int(state.step_count) + 1
    assert float(
        jnp.linalg.norm(result.state.head_params.weights[0] - state.head_params.weights[0])
    ) > 0.0


def test_nonzero_discount_rejects_nonfinite_next_then_recovers_under_jit() -> None:
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.9,)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(2, jax.random.key(20))
    observation = jnp.array([1.0, -0.5], dtype=jnp.float32)
    cumulants = jnp.array([1.0], dtype=jnp.float32)
    rhos = jnp.ones(1, dtype=jnp.float32)
    discounts = jnp.array([0.9], dtype=jnp.float32)

    rejected = learner.update_with_ratios_and_discounts(
        state,
        observation,
        cumulants,
        jnp.array([jnp.inf, 0.0], dtype=jnp.float32),
        rhos,
        discounts,
    )
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.head_updates_applied, jnp.array([False]))
    chex.assert_trees_all_close(rejected.state, state)
    chex.assert_trees_all_close(rejected.next_predictions, jnp.zeros(1))
    chex.assert_trees_all_close(rejected.td_targets, jnp.zeros(1))

    recovered = learner.update_with_ratios_and_discounts(
        rejected.state,
        observation,
        cumulants,
        jnp.array([0.25, -0.25], dtype=jnp.float32),
        rhos,
        discounts,
    )
    assert bool(recovered.update_applied)
    chex.assert_trees_all_equal(recovered.head_updates_applied, jnp.array([True]))
    chex.assert_tree_all_finite(recovered.state)
    assert int(recovered.state.step_count) == int(state.step_count) + 1


def test_mixed_discounts_isolate_nonfinite_next_to_consuming_head() -> None:
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0, 0.9)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(2, jax.random.key(21))
    next_observation = jnp.array([jnp.inf, 0.0], dtype=jnp.float32)
    assert bool(jnp.all(~jnp.isfinite(learner.predict(state, next_observation))))

    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.array([1.0, -0.5], dtype=jnp.float32),
        jnp.array([1.25, 0.5], dtype=jnp.float32),
        next_observation,
        jnp.ones(2, dtype=jnp.float32),
        jnp.array([0.0, 0.9], dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(
        result.head_updates_applied,
        jnp.array([True, False]),
    )
    chex.assert_trees_all_close(result.td_targets, jnp.array([1.25, 0.0]))
    chex.assert_trees_all_close(result.next_predictions, jnp.zeros(2))
    assert float(
        jnp.linalg.norm(result.state.head_params.weights[0] - state.head_params.weights[0])
    ) > 0.0
    chex.assert_trees_all_close(
        result.state.head_params.weights[1],
        state.head_params.weights[1],
    )
    chex.assert_trees_all_close(
        result.state.head_params.biases[1],
        state.head_params.biases[1],
    )
    chex.assert_trees_all_close(
        result.state.head_traces[1],
        state.head_traces[1],
    )
    chex.assert_trees_all_close(
        result.state.head_optimizer_states[1],
        state.head_optimizer_states[1],
    )


def test_terminal_lifetime_counter_rejects_entire_update_under_jit() -> None:
    """An exhausted outer clock cannot commit parameters or report success."""
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0,)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(2, jax.random.key(17)).replace(
        step_count=jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
        step_words=jnp.full(
            (2,), jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32
        ),
    )

    rejected = learner.update_with_ratios(
        state,
        jnp.array([1.0, -0.5], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
    )

    assert not bool(rejected.update_applied)
    chex.assert_trees_all_close(rejected.state, state)
    chex.assert_trees_all_equal(
        rejected.head_updates_applied, jnp.array([False])
    )
    chex.assert_trees_all_close(rejected.predictions, jnp.zeros(1))
    chex.assert_trees_all_close(rejected.td_errors, jnp.zeros(1))
    chex.assert_trees_all_close(rejected.per_demon_metrics, jnp.zeros((1, 6)))
    assert float(rejected.trunk_bounding_metric) == 0.0


def test_invalid_lifetime_counter_rejects_then_repaired_state_recovers() -> None:
    """Counter inconsistency is transient once an authorized state is restored."""
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0,)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        sparsity=0.0,
        use_layer_norm=False,
    )
    initial = learner.init(2, jax.random.key(18))
    invalid = initial.replace(
        step_words=jnp.array([0, 1], dtype=jnp.uint32),
    )
    inputs = (
        jnp.array([1.0, -0.5], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
    )

    rejected = learner.update_with_ratios(invalid, *inputs)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_close(rejected.state, invalid)

    recovered = learner.update_with_ratios(initial, *inputs)
    assert bool(recovered.update_applied)
    assert int(recovered.state.step_count) == 1
    chex.assert_trees_all_equal(
        recovered.state.step_words, jnp.array([0, 1], dtype=jnp.uint32)
    )


def test_probability_api_matches_explicit_ratios() -> None:
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0, 0.0)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.05),
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(3))
    obs = jnp.array([1.0, -1.0], dtype=jnp.float32)
    cumulants = jnp.array([1.0, -1.0], dtype=jnp.float32)
    next_obs = jnp.zeros(2, dtype=jnp.float32)

    direct = learner.update_with_ratios(
        state,
        obs,
        cumulants,
        next_obs,
        jnp.array([0.5, 2.0], dtype=jnp.float32),
    )
    probabilistic = learner.update_with_probabilities(
        state,
        obs,
        cumulants,
        next_obs,
        jnp.array([0.25, 1.0], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    )

    chex.assert_trees_all_close(direct.clipped_rhos, probabilistic.clipped_rhos)
    chex.assert_trees_all_close(
        direct.state.head_params,
        probabilistic.state.head_params,
        atol=1e-6,
    )


def test_scan_loop_shapes_and_finite_state() -> None:
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0, 0.5), lamdas=(0.0, 0.4)),
        hidden_sizes=(5,),
        optimizer=LMS(step_size=0.01),
        ratio_clip=2.0,
    )
    state = learner.init(3, jax.random.key(4))
    observations = jnp.ones((6, 3), dtype=jnp.float32)
    next_observations = jnp.roll(observations, shift=-1, axis=0)
    cumulants = jnp.stack(
        [
            jnp.linspace(0.0, 1.0, 6),
            jnp.linspace(1.0, 0.0, 6),
        ],
        axis=1,
    ).astype(jnp.float32)
    rhos = jnp.ones((6, 2), dtype=jnp.float32) * jnp.array([1.0, 1.5])

    result = run_off_policy_horde_learning_loop(
        learner,
        state,
        observations,
        cumulants,
        next_observations,
        rhos,
    )

    chex.assert_shape(result.per_demon_metrics, (6, 2, 6))
    chex.assert_shape(result.td_errors, (6, 2))
    chex.assert_shape(result.clipped_rhos, (6, 2))
    chex.assert_tree_all_finite(result.state.trunk_params)
    chex.assert_tree_all_finite(result.state.head_params)


def test_off_policy_positive_control_learns_target_action_value() -> None:
    rng = np.random.default_rng(5)
    actions = rng.integers(0, 2, size=240)
    observations = jnp.ones((240, 1), dtype=jnp.float32)
    next_observations = jnp.ones((240, 1), dtype=jnp.float32)
    cumulants = jnp.asarray((actions == 1).astype(np.float32)).reshape(-1, 1)
    target_rhos = jnp.asarray(
        np.where(actions == 1, 2.0, 0.0).astype(np.float32)
    ).reshape(-1, 1)
    no_is_rhos = jnp.ones((240, 1), dtype=jnp.float32)

    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0,)),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.02),
        ratio_clip=10.0,
        sparsity=0.0,
    )
    initial_state = learner.init(1, jax.random.key(6))

    off_policy = run_off_policy_horde_learning_loop(
        learner,
        initial_state,
        observations,
        cumulants,
        next_observations,
        target_rhos,
    )
    no_is = run_off_policy_horde_learning_loop(
        learner,
        initial_state,
        observations,
        cumulants,
        next_observations,
        no_is_rhos,
    )

    target_pred = float(learner.predict(off_policy.state, jnp.ones(1))[0])
    behavior_pred = float(learner.predict(no_is.state, jnp.ones(1))[0])
    assert target_pred > 0.85
    assert target_pred > behavior_pred + 0.25


def test_nonlinear_shared_gtd_horde_updates_secondary_and_trunk() -> None:
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(0.8, 0.8)),
        hidden_size=4,
        primary_step_size=0.002,
        secondary_step_size=1e-5,
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(7))
    before_trunk = state.trunk_w
    obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    next_obs = jnp.array([0.0, 1.0], dtype=jnp.float32)

    result = learner.update_with_ratios_and_discounts(
        state,
        obs,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        next_obs,
        jnp.array([2.0, 0.0], dtype=jnp.float32),
        jnp.array([0.8, 0.8], dtype=jnp.float32),
    )

    chex.assert_shape(result.predictions, (2,))
    chex.assert_shape(result.correction_norms, (2,))
    chex.assert_tree_all_finite(result.state)
    assert float(jnp.linalg.norm(result.state.trunk_w - before_trunk)) > 0.0
    assert float(result.secondary_norms[0]) > 0.0
    assert float(result.secondary_norms[1]) == pytest.approx(0.0)


def test_nonlinear_shared_gtd_horde_two_state_positive_control() -> None:
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(0.8, 0.8)),
        hidden_size=8,
        primary_step_size=0.002,
        secondary_step_size=1e-5,
        ratio_clip=10.0,
    )
    rng = np.random.default_rng(9)
    steps = 3000
    states = np.empty(steps, dtype=np.int32)
    actions = np.empty(steps, dtype=np.int32)
    rewards = np.empty(steps, dtype=np.float32)
    next_states = np.empty(steps, dtype=np.int32)
    s = 0
    for t in range(steps):
        a = int(rng.integers(0, 2))
        states[t] = s
        actions[t] = a
        rewards[t] = 1.0 if s == a else 0.0
        next_states[t] = a
        s = a
    observations = jnp.asarray(np.eye(2, dtype=np.float32)[states])
    next_observations = jnp.asarray(np.eye(2, dtype=np.float32)[next_states])
    cumulants = jnp.repeat(jnp.asarray(rewards)[:, None], 2, axis=1)
    rhos = np.zeros((steps, 2), dtype=np.float32)
    rhos[actions == 0, 0] = 2.0
    rhos[actions == 1, 1] = 2.0
    rhos_jnp = jnp.asarray(rhos)
    discounts = jnp.full((steps, 2), 0.8, dtype=jnp.float32)

    def step(carry, xs):  # type: ignore[no-untyped-def]
        obs, cums, next_obs, rho, discount = xs
        update = learner.update_with_ratios_and_discounts(
            carry, obs, cums, next_obs, rho, discount
        )
        return update.state, update

    initial = learner.init(2, jax.random.key(9))
    final_state, updates = jax.lax.scan(
        step,
        initial,
        (observations, cumulants, next_observations, rhos_jnp, discounts),
    )
    predictions = jax.vmap(lambda x: learner.predict(final_state, x))(
        jnp.eye(2, dtype=jnp.float32)
    )
    target = jnp.array([[5.0, 4.0], [4.0, 5.0]], dtype=jnp.float32)

    assert float(jnp.mean(jnp.abs(predictions - target))) < 0.8
    assert float(jnp.mean(updates.secondary_norms[-100:])) > 0.0
    assert float(jnp.mean(updates.correction_norms[-100:])) > 0.0


def test_nonlinear_shared_gtd_zero_discount_does_not_multiply_inf_next() -> None:
    """gamma=0 * inf next-grad is 0*inf = NaN in the TDC correction."""
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(0.0,)),
        hidden_size=2,
        primary_step_size=0.01,
        secondary_step_size=0.01,
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(0))
    next_obs = jnp.array([jnp.inf, 0.0], dtype=jnp.float32)
    raw = jnp.asarray(0.0, dtype=jnp.float32) * next_obs[0]
    assert not bool(jnp.isfinite(raw))

    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([3.0], dtype=jnp.float32),
        next_obs,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(result.state)
    chex.assert_trees_all_close(result.td_targets, jnp.array([3.0], dtype=jnp.float32))
    chex.assert_tree_all_finite(result.correction_norms)


def test_zero_utility_decay_does_not_multiply_inf_utility() -> None:
    """utility_decay=0 times leftover inf hidden-unit utility is NaN."""
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0, 0.0)),
        hidden_sizes=(4,),
        optimizer=LMS(step_size=0.01),
        utility_decay=0.0,
        sparsity=0.0,
    )
    state = learner.init(3, jax.random.key(0))
    state = state.replace(  # type: ignore[attr-defined]
        hidden_unit_utilities=tuple(
            jnp.full_like(utility, jnp.inf) for utility in state.hidden_unit_utilities
        )
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = learner.update_with_ratios(
        state,
        jnp.array([0.2, -0.1, 0.4], dtype=jnp.float32),
        jnp.array([1.0, -0.5], dtype=jnp.float32),
        jnp.array([0.0, 0.1, 0.2], dtype=jnp.float32),
        jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    for utility in result.state.hidden_unit_utilities:
        chex.assert_tree_all_finite(utility)


def test_replacing_zero_grad_does_not_multiply_inf_trunk_trace() -> None:
    """Replacing traces used old * 0.0 on silent coordinates, which is 0*inf."""
    learner = OffPolicyHordeLearner(
        _spec(gammas=(0.0, 0.0)),
        hidden_sizes=(3,),
        optimizer=LMS(step_size=0.01),
        trace_mode=TraceMode.REPLACING,
        sparsity=0.0,
    )
    state = learner.init(2, jax.random.key(1))
    state = state.replace(  # type: ignore[attr-defined]
        trunk_traces=tuple(jnp.full_like(trace, jnp.inf) for trace in state.trunk_traces)
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = learner.update_with_ratios(
        state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        jnp.array([0.5, -0.25], dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    for trace in result.state.trunk_traces:
        chex.assert_tree_all_finite(trace)
