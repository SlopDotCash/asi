"""Tests for nonlinear off-policy Horde backend."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core import off_policy_horde as off_policy_horde_module
from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPState,
    measure_multi_head_mlp_state_nbytes,
)
from alberta_framework.core.off_policy_horde import (
    NonlinearSharedGTDHordeLearner,
    OffPolicyHordeLearner,
    OffPolicyHordeUpdateResult,
    run_off_policy_horde_learning_loop,
    run_off_policy_horde_learning_loop_batched,
)
from alberta_framework.core.optimizers import LMS, AGCBounding, ObGDBounding
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


@pytest.mark.parametrize("mutation", ["missing_ratio_clip", "wrong_type"])
def test_config_adoption_requires_exact_serialized_schema(mutation: str) -> None:
    learner = OffPolicyHordeLearner(
        _spec(),
        hidden_sizes=(8,),
        optimizer=LMS(step_size=0.03),
        ratio_clip=2.0,
    )
    config = learner.to_config()
    if mutation == "missing_ratio_clip":
        config.pop("ratio_clip")
    else:
        config["type"] = "MultiHeadMLPLearner"

    with pytest.raises(ValueError, match="OffPolicyHordeLearner config"):
        OffPolicyHordeLearner.from_config(config)


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
    assert float(jnp.linalg.norm(poisoned.state.head_params.weights[1] - before_w1)) > 0.0
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


@pytest.mark.parametrize("bounded", ["obgd", "agc_clips_nothing"])
def test_bounder_metric_never_scales_traces(bounded: str) -> None:
    """Bounding rescales the applied step only; traces decay by gamma*lambda.

    ``Bounder.bound`` returns ``(steps, metric)`` where the metric is for
    reporting (ObGD's step scale, AGC's clipped-unit fraction, 0.0 when
    nothing is clipped). ``OffPolicyHordeLearner`` multiplied it into the
    stored head traces, so AGC that clips nothing zeroed TD(lambda) and
    ObGD attenuated traces the ObGD rule does not scale. A linear demon
    with ``rho=1`` must match the textbook semi-gradient TD(lambda)
    reference (ObGD scales only the applied step).
    """
    alpha, gamma, lamda, kappa, dim, n_steps = 0.05, 0.9, 0.8, 2.0, 3, 8
    spec = _spec(gammas=(gamma,), lamdas=(lamda,))
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(n_steps + 1, dim)).astype(np.float32)
    cumulants = (3.0 * rng.normal(size=n_steps)).astype(np.float32)
    bounder = ObGDBounding(kappa=kappa) if bounded == "obgd" else AGCBounding(clip_factor=1e6)
    learner = OffPolicyHordeLearner(
        spec,
        hidden_sizes=(),
        optimizer=LMS(step_size=alpha),
        bounder=bounder,
        sparsity=0.0,
        use_layer_norm=False,
        ratio_clip=10.0,
        trace_ratio_clip=10.0,
    )
    state = learner.init(dim, jax.random.key(0))
    w = np.asarray(state.head_params.weights[0], dtype=np.float64).reshape(-1)
    b = float(np.asarray(state.head_params.biases[0]).reshape(-1)[0])
    z_w, z_b = np.zeros(dim), 0.0
    min_scale = 1.0
    rhos = jnp.ones(1, dtype=jnp.float32)
    for t in range(n_steps):
        state = learner.update_with_ratios(
            state,
            jnp.asarray(xs[t]),
            jnp.asarray(cumulants[t : t + 1]),
            jnp.asarray(xs[t + 1]),
            rhos,
        ).state
        x, x_next = xs[t].astype(np.float64), xs[t + 1].astype(np.float64)
        delta = cumulants[t] + gamma * (w @ x_next + b) - (w @ x + b)
        z_w, z_b = gamma * lamda * z_w + x, gamma * lamda * z_b + 1.0
        scale = 1.0
        if bounded == "obgd":
            total = alpha * (np.abs(z_w).sum() + abs(z_b))
            scale = 1.0 / max(kappa * max(abs(delta), 1.0) * total, 1.0)
        min_scale = min(min_scale, scale)
        w, b = w + scale * alpha * delta * z_w, b + scale * alpha * delta * z_b

    if bounded == "obgd":
        assert min_scale < 0.9  # the bound was active
    else:
        traces = np.asarray(state.head_traces[0][0], dtype=np.float64).reshape(-1)
        assert float(np.abs(traces).sum()) > 0.0
    coded_w = np.asarray(state.head_params.weights[0], dtype=np.float64).reshape(-1)
    coded_b = float(np.asarray(state.head_params.biases[0]).reshape(-1)[0])
    np.testing.assert_allclose(coded_w, w, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(coded_b, b, rtol=0.0, atol=1e-4)


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
    assert (
        float(jnp.linalg.norm(result.state.head_params.weights[0] - state.head_params.weights[0]))
        > 0.0
    )


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
    assert (
        float(jnp.linalg.norm(result.state.head_params.weights[0] - state.head_params.weights[0]))
        > 0.0
    )
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


# --- Incoming-discount trace decay (state-carried ``previous_discounts``) ---
#
# ``OffPolicyHordeState`` is reached through the module attribute so this file
# still collects against a production module that lacks the class; the
# behavioural tests below then fail on their assertions rather than at import.


def _history_learner(
    mode: TraceMode = TraceMode.ACCUMULATING, n_demons: int = 1
) -> OffPolicyHordeLearner:
    return OffPolicyHordeLearner(
        _spec(gammas=(0.9,) * n_demons, lamdas=(0.8,) * n_demons),
        hidden_sizes=(),
        optimizer=LMS(step_size=0.1),
        ratio_clip=10.0,
        trace_ratio_clip=10.0,
        trace_mode=mode,
    )


def _zeroed_history_state(learner: OffPolicyHordeLearner) -> Any:
    state = learner.init(3, jax.random.key(270, impl="threefry2x32"))
    return state.replace(head_params=jax.tree.map(jnp.zeros_like, state.head_params))


def _history_step(
    learner: OffPolicyHordeLearner,
    state: Any,
    index: int,
    rewards: list[float],
    discounts: list[float],
    rhos: list[float] | None = None,
) -> OffPolicyHordeUpdateResult:
    """One-hot ``e_index -> e_{index+1}`` transition with explicit discounts."""
    return learner.update_with_ratios_and_discounts(
        state,
        jnp.eye(3, dtype=jnp.float32)[index],
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32)[(index + 1) % 3],
        jnp.asarray(rhos if rhos is not None else [1.0] * learner.n_demons, dtype=jnp.float32),
        jnp.asarray(discounts, dtype=jnp.float32),
    )


def _assert_learning_state_equal(left: Any, right: Any) -> None:
    for name in dict(left):
        if name in {"birth_timestamp", "uptime_s"}:
            continue
        for a, b in zip(
            jax.tree.leaves(getattr(left, name)), jax.tree.leaves(getattr(right, name)), strict=True
        ):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode", list(TraceMode))
@pytest.mark.parametrize("incoming", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("outgoing", [0.0, 0.25, 1.0])
def test_delayed_credit_uses_incoming_discount(
    mode: TraceMode, incoming: float, outgoing: float
) -> None:
    """The rewarded step decays the carried trace by the discount that entered it."""
    learner = _history_learner(mode)
    first = _history_step(learner, _zeroed_history_state(learner), 0, [0.0], [incoming], [2.0])
    result = _history_step(learner, first.state, 1, [1.0], [outgoing], [0.5])
    assert bool(first.update_applied) and bool(result.update_applied)
    # delta_1 = 1, so w = alpha * rho_1 * (gamma_in * lambda * rho_0 * phi_0 + phi_1);
    # the outgoing discount of the rewarded transition must not appear.
    expected = np.array([[0.1 * 0.5 * incoming * 0.8 * 2.0, 0.05, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(result.state.head_params.weights[0], expected, atol=1e-7)
    np.testing.assert_array_equal(result.state.previous_discounts, [outgoing])


@pytest.mark.parametrize("mode", list(TraceMode))
def test_terminal_credit_does_not_cross_into_next_episode(mode: TraceMode) -> None:
    learner = _history_learner(mode)
    first = _history_step(learner, _zeroed_history_state(learner), 0, [0.0], [0.5])
    terminal = _history_step(learner, first.state, 1, [1.0], [0.0])
    following = _history_step(learner, terminal.state, 2, [1.0], [0.9])
    np.testing.assert_array_equal(
        following.state.head_params.weights[0][:, :2],
        terminal.state.head_params.weights[0][:, :2],
    )


@pytest.mark.parametrize("skipped_reward", [np.nan, np.inf])
def test_inactive_or_rejected_head_preserves_its_own_discount(skipped_reward: float) -> None:
    learner = _history_learner(n_demons=2)
    first = _history_step(learner, _zeroed_history_state(learner), 0, [0.0, 0.0], [0.5, 0.9])
    skipped = _history_step(learner, first.state, 1, [skipped_reward, 0.0], [0.0, 0.25])
    np.testing.assert_array_equal(skipped.head_updates_applied, [False, True])
    np.testing.assert_array_equal(skipped.state.previous_discounts, [0.5, 0.25])
    np.testing.assert_array_equal(skipped.state.head_traces[0][0], first.state.head_traces[0][0])
    resumed = _history_step(learner, skipped.state, 2, [1.0, 1.0], [0.0, 0.0])
    # Head 0 resumes its last accepted trace under its own carried discount;
    # head 1 decays the two-step trace by its newer history.
    np.testing.assert_allclose(resumed.state.head_params.weights[0], [[0.04, 0.0, 0.1]], atol=1e-7)
    np.testing.assert_allclose(
        resumed.state.head_params.weights[1], [[0.0144, 0.02, 0.1]], atol=1e-7
    )


def test_rejected_transition_preserves_carried_discount_for_retry() -> None:
    """A globally rejected (non-finite) transition leaves the history intact for retry."""
    learner = _history_learner(n_demons=2)
    first = _history_step(learner, _zeroed_history_state(learner), 0, [0.0, 0.0], [0.5, 0.9])
    rejected = learner.update_with_ratios_and_discounts(
        first.state,
        jnp.full(3, jnp.inf, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
        jnp.zeros(3, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
    )
    assert not bool(rejected.update_applied)
    _assert_learning_state_equal(rejected.state, first.state)
    np.testing.assert_array_equal(
        rejected.state.previous_discounts, np.array([0.5, 0.9], dtype=np.float32)
    )
    retried = _history_step(learner, rejected.state, 1, [1.0, 1.0], [0.0, 0.0])
    reference = _history_step(learner, first.state, 1, [1.0, 1.0], [0.0, 0.0])
    assert bool(retried.update_applied)
    _assert_learning_state_equal(retried.state, reference.state)
    # An all-inactive accepted step also leaves every head's history alone.
    inactive = _history_step(learner, first.state, 1, [np.nan, np.nan], [0.0, 0.0])
    assert bool(inactive.update_applied)
    np.testing.assert_array_equal(
        inactive.state.previous_discounts, np.array([0.5, 0.9], dtype=np.float32)
    )
    assert int(inactive.state.step_count) == int(first.state.step_count) + 1


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_numeric_history_rejects_atomically(bad: float) -> None:
    learner = _history_learner()
    state = _zeroed_history_state(learner).replace(
        previous_discounts=jnp.array([bad], dtype=jnp.float32)
    )
    rejected = _history_step(learner, state, 0, [1.0], [0.0])
    assert not bool(rejected.update_applied)
    _assert_learning_state_equal(rejected.state, state)


@pytest.mark.parametrize(
    "bad",
    [None, jnp.array(0.5, dtype=jnp.float32), jnp.ones(2), jnp.ones(1, dtype=jnp.int32)],
)
def test_noncanonical_history_fails_closed_before_tracing(bad: object) -> None:
    learner = _history_learner()
    state = _zeroed_history_state(learner).replace(previous_discounts=bad)
    with pytest.raises((TypeError, ValueError), match="previous_discounts"):
        _history_step(learner, state, 0, [1.0], [0.0])


def test_legacy_state_requires_explicit_adoption_and_charges_one_float_per_demon() -> None:
    """Legacy ``MultiHeadMLPState`` cannot resume silently; adoption is explicit.

    The only persistent growth is the float32 history vector: 4 bytes per demon.
    """
    learner = _history_learner(n_demons=2)
    key = jax.random.key(270, impl="threefry2x32")
    legacy = learner.learner.init(3, key)
    assert type(legacy) is MultiHeadMLPState
    with pytest.raises(TypeError, match="previous_discounts"):
        _history_step(learner, legacy, 0, [0.0, 0.0], [0.0, 0.0])
    with pytest.raises(TypeError, match="previous_discounts"):
        off_policy_horde_module.OffPolicyHordeState(**dict(legacy))
    adopted = off_policy_horde_module.OffPolicyHordeState(
        **dict(legacy), previous_discounts=jnp.ones(2, dtype=jnp.float32)
    )
    fresh = learner.init(3, key)
    assert isinstance(fresh, off_policy_horde_module.OffPolicyHordeState)
    assert isinstance(fresh, MultiHeadMLPState)
    _assert_learning_state_equal(adopted, fresh)
    assert fresh.previous_discounts.dtype == jnp.float32
    assert (
        measure_multi_head_mlp_state_nbytes(fresh)
        == measure_multi_head_mlp_state_nbytes(legacy) + 4 * learner.n_demons
    )
    assert len(jax.tree.leaves(fresh)) == len(jax.tree.leaves(legacy)) + 1
    result = _history_step(learner, adopted, 0, [0.0, 0.0], [0.0, 0.0])
    assert bool(result.update_applied)


@pytest.mark.parametrize("incoming, accepted", [(0.0, True), (0.5, False)])
def test_zero_outgoing_discount_does_not_hide_required_poisoned_trace(
    incoming: float, accepted: bool
) -> None:
    learner = _history_learner()
    first = _history_step(learner, _zeroed_history_state(learner), 0, [0.0], [incoming])
    state = first.state.replace(
        head_traces=((jnp.full((1, 3), jnp.inf, dtype=jnp.float32), jnp.ones(1)),)
    )
    result = _history_step(learner, state, 1, [1.0], [0.0])
    assert bool(result.update_applied) == accepted
    if accepted:
        np.testing.assert_array_equal(result.state.head_traces[0][0], [[0.0, 1.0, 0.0]])
    else:
        _assert_learning_state_equal(result.state, state)


def test_pickle_and_public_scan_preserve_discount_history() -> None:
    learner = _history_learner()
    initial = _zeroed_history_state(learner)
    first = _history_step(learner, initial, 0, [0.0], [0.5])
    restored = pickle.loads(pickle.dumps(first.state))
    assert isinstance(restored, off_policy_horde_module.OffPolicyHordeState)
    np.testing.assert_array_equal(restored.previous_discounts, [0.5])
    expected = _history_step(learner, first.state, 1, [1.0], [0.0])
    actual = _history_step(learner, restored, 1, [1.0], [0.0])
    _assert_learning_state_equal(actual.state, expected.state)
    scanned = run_off_policy_horde_learning_loop(
        learner,
        initial,
        jnp.eye(3, dtype=jnp.float32)[:2],
        jnp.array([[0.0], [1.0]], dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32)[1:],
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.array([[0.5], [0.0]], dtype=jnp.float32),
    )
    _assert_learning_state_equal(scanned.state, expected.state)
    np.testing.assert_array_equal(scanned.updates_applied, [True, True])


def test_checkpoint_round_trip_carries_history_and_legacy_template_fails_closed(
    tmp_path: Path,
) -> None:
    learner = _history_learner()
    key = jax.random.key(270, impl="threefry2x32")
    first = _history_step(learner, _zeroed_history_state(learner), 0, [0.0], [0.5])
    save_checkpoint(first.state, tmp_path / "history")
    loaded, _ = load_checkpoint(learner.init(3, key), tmp_path / "history")
    assert isinstance(loaded, off_policy_horde_module.OffPolicyHordeState)
    np.testing.assert_array_equal(loaded.previous_discounts, [0.5])
    expected = _history_step(learner, first.state, 1, [1.0], [0.0])
    actual = _history_step(learner, loaded, 1, [1.0], [0.0])
    _assert_learning_state_equal(actual.state, expected.state)
    # The generic multi-head template has one leaf fewer: refuse rather than misread.
    with pytest.raises(ValueError, match="leaf count"):
        load_checkpoint(learner.learner.init(3, key), tmp_path / "history")


def test_batched_public_scan_keeps_history_in_its_state() -> None:
    learner = _history_learner()
    keys = jax.random.split(jax.random.key(270, impl="threefry2x32"), 2)
    result = run_off_policy_horde_learning_loop_batched(
        learner,
        jnp.eye(3, dtype=jnp.float32)[:2],
        jnp.zeros((2, 1), dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32)[1:],
        jnp.ones((2, 1), dtype=jnp.float32),
        keys,
        jnp.array([[0.5], [0.0]], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(result.state.previous_discounts, [[0.0], [0.0]])


def test_history_bytes_are_preflighted_before_wrapped_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three live float32 history copies (12 bytes/demon) join the int32 envelope."""
    learner = _history_learner(n_demons=2)
    monkeypatch.setattr(
        off_policy_horde_module,
        "_multi_head_update_working_set_bytes",
        lambda *args: 2**31 - 20,
    )

    def forbidden(*args: object) -> None:
        pytest.fail("oversized history envelope reached allocation")

    monkeypatch.setattr(learner.learner, "init", forbidden)
    with pytest.raises(ValueError, match="working set"):
        learner.init(3, jax.random.key(270, impl="threefry2x32"))


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
        step_words=jnp.full((2,), jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32),
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
    chex.assert_trees_all_equal(rejected.head_updates_applied, jnp.array([False]))
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
    chex.assert_trees_all_equal(recovered.state.step_words, jnp.array([0, 1], dtype=jnp.uint32))


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
    target_rhos = jnp.asarray(np.where(actions == 1, 2.0, 0.0).astype(np.float32)).reshape(-1, 1)
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
        update = learner.update_with_ratios_and_discounts(carry, obs, cums, next_obs, rho, discount)
        return update.state, update

    initial = learner.init(2, jax.random.key(9))
    final_state, updates = jax.lax.scan(
        step,
        initial,
        (observations, cumulants, next_observations, rhos_jnp, discounts),
    )
    predictions = jax.vmap(lambda x: learner.predict(final_state, x))(jnp.eye(2, dtype=jnp.float32))
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


def test_nonlinear_shared_gtd_horde_integer_validation() -> None:
    spec = _spec()

    with pytest.raises(ValueError, match="hidden_size"):
        NonlinearSharedGTDHordeLearner(horde_spec=spec, hidden_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_size"):
        NonlinearSharedGTDHordeLearner(horde_spec=spec, hidden_size=16.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_size"):
        NonlinearSharedGTDHordeLearner(horde_spec=spec, hidden_size=0)

    learner = NonlinearSharedGTDHordeLearner(horde_spec=spec, hidden_size=np.int32(16))
    assert learner._hidden_size == 16
    assert type(learner._hidden_size) is int

    key = jax.random.key(0)
    with pytest.raises(ValueError, match="feature_dim"):
        learner.init(feature_dim=True, key=key)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        learner.init(feature_dim=4.5, key=key)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        learner.init(feature_dim=0, key=key)

    state = learner.init(feature_dim=np.int32(4), key=key)
    assert state.trunk_w.shape == (16, 4)
    with pytest.raises(ValueError, match="typed scalar"):
        learner.init(feature_dim=4, key=jax.random.PRNGKey(0))


def test_nonlinear_shared_gtd_horde_complete_scalar_and_schema_contract() -> None:
    calls = 0

    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile hook ran")

    with pytest.raises(ValueError, match="finite real scalar"):
        NonlinearSharedGTDHordeLearner(_spec(), primary_step_size=HostileFloat(0.1))
    assert calls == 0

    learner = NonlinearSharedGTDHordeLearner(
        _spec(),
        hidden_size=np.uint16(4),
        primary_step_size=np.float64(0.01),
    )
    config = learner.to_config()
    restored = NonlinearSharedGTDHordeLearner.from_config(config)
    assert restored.to_config() == config
    with pytest.raises(ValueError, match="schema"):
        NonlinearSharedGTDHordeLearner.from_config({**config, "unknown": 1})
    with pytest.raises(ValueError, match="type"):
        NonlinearSharedGTDHordeLearner.from_config({**config, "type": "wrong"})


def test_nonlinear_shared_gtd_horde_preflights_aggregate_state_bytes() -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(), hidden_size=16)
    with pytest.raises(ValueError, match="persistent state bytes"):
        learner.init(20_000_000, jax.random.key(0))


def test_nonlinear_shared_gtd_horde_outer_counter_saturates() -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(gammas=(0.0,)), hidden_size=2)
    state = learner.init(2, jax.random.key(0)).replace(
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32)
    )
    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == 2**31 - 1


def test_nonlinear_shared_gtd_horde_runtime_metadata_is_hostile_safe() -> None:
    calls = 0

    class HostileLeaf:
        @property
        def shape(self):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            raise AssertionError("shape hook ran")

    learner = NonlinearSharedGTDHordeLearner(_spec(), hidden_size=2)
    malformed = learner.init(2, jax.random.key(0)).replace(trunk_w=HostileLeaf())
    with pytest.raises(TypeError, match="state.trunk_w"):
        learner.predict(malformed, jnp.ones(2, dtype=jnp.float32))
    assert calls == 0

    state = learner.init(2, jax.random.key(0))
    with pytest.raises(TypeError, match="observation"):
        learner.predict(state, jnp.ones(2, dtype=jnp.float16))
    with pytest.raises(ValueError, match="shape"):
        learner.predict(state, jnp.ones(3, dtype=jnp.float32))


def test_nonlinear_shared_gtd_horde_terminal_next_and_ratio_domains() -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(gammas=(0.0,)), hidden_size=2)
    state = learner.init(2, jax.random.key(0))
    terminal = learner.update_with_ratios_and_discounts(
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.full(2, jnp.inf, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )
    assert bool(terminal.update_applied)
    chex.assert_trees_all_equal(terminal.next_predictions, jnp.zeros(1, dtype=jnp.float32))

    rejected = learner.update_with_ratios_and_discounts(
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
        -jnp.ones(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, state)


def test_nonlinear_shared_gtd_init_rejects_nonfinite_generated_state() -> None:
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(0.0,)),
        hidden_size=2,
        init_scale=float(jnp.finfo(jnp.float32).max),
    )
    with pytest.raises(ValueError, match="initialized nonlinear Horde state"):
        learner.init(2, jax.random.key(0))


def test_nonlinear_shared_gtd_norms_are_stable_for_large_finite_state() -> None:
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(0.0,)), hidden_size=2, secondary_step_size=1e-5
    )
    state = learner.init(2, jax.random.key(0)).replace(
        secondary_trunk_w=jnp.full((1, 2, 2), 1e30, dtype=jnp.float32),
        secondary_trunk_b=jnp.full((1, 2), 1e30, dtype=jnp.float32),
        secondary_head_w=jnp.full((1, 2), 1e30, dtype=jnp.float32),
        secondary_head_b=jnp.full((1,), 1e30, dtype=jnp.float32),
    )
    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.zeros(2, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    chex.assert_tree_all_finite(result.correction_norms)
    chex.assert_tree_all_finite(result.secondary_norms)


def test_nonlinear_shared_gtd_rejection_neutralizes_complete_result() -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(gammas=(1.0,)), hidden_size=2)
    state = learner.init(2, jax.random.key(0))
    poisoned = state.replace(trunk_w=jnp.full_like(state.trunk_w, jnp.inf))
    result = learner.update_with_ratios_and_discounts(
        poisoned,
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    for value in (
        result.predictions,
        result.next_predictions,
        result.td_targets,
        result.td_errors,
        result.clipped_rhos,
        result.correction_norms,
        result.secondary_norms,
    ):
        chex.assert_trees_all_equal(value, jnp.zeros_like(value))


def test_off_policy_horde_ratio_clip_scalars_reject_booleans_and_nans() -> None:
    spec = _spec(gammas=(1.0,))

    # ratio_clip
    with pytest.raises(ValueError, match="ratio_clip"):
        OffPolicyHordeLearner(spec, ratio_clip=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ratio_clip"):
        OffPolicyHordeLearner(spec, ratio_clip=float("nan"))
    # +inf stays legal: it is the documented no-clip sentinel the Baird
    # counterexample tests rely on.
    assert OffPolicyHordeLearner(spec, ratio_clip=float("inf")).ratio_clip == float("inf")
    with pytest.raises(ValueError, match="ratio_clip"):
        OffPolicyHordeLearner(spec, ratio_clip=0.0)

    # trace_ratio_clip
    with pytest.raises(ValueError, match="trace_ratio_clip"):
        OffPolicyHordeLearner(spec, trace_ratio_clip=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trace_ratio_clip"):
        OffPolicyHordeLearner(spec, trace_ratio_clip=float("nan"))
    with pytest.raises(ValueError, match="trace_ratio_clip"):
        OffPolicyHordeLearner(spec, trace_ratio_clip=0.0)

    # min_behavior_probability
    with pytest.raises(ValueError, match="min_behavior_probability"):
        OffPolicyHordeLearner(spec, min_behavior_probability=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_behavior_probability"):
        OffPolicyHordeLearner(spec, min_behavior_probability=float("nan"))
    with pytest.raises(ValueError, match="min_behavior_probability"):
        OffPolicyHordeLearner(spec, min_behavior_probability=0.0)

    # Valid construction
    learner = OffPolicyHordeLearner(
        spec, ratio_clip=2, trace_ratio_clip=1.0, min_behavior_probability=1e-6
    )
    assert learner._ratio_clip == 2.0
    assert learner._trace_ratio_clip == 1.0
    assert learner._min_behavior_probability == 1e-6
