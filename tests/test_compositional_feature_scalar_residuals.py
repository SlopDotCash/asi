"""Residual scalar and recursive-resource contracts for compositional features."""

from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    GENERATION_ROBUST_RECURSIVE,
    CompositionalFeatureLearner,
    FiniteCandidateSelector,
    FiniteCandidateSelectorState,
)


class _HostileFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
        raise RuntimeError("ratio hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


def _learner(**overrides: Any) -> CompositionalFeatureLearner:
    values: dict[str, Any] = {"n_features": 6, "n_tasks": 2, "candidate_count": 2}
    values.update(overrides)
    return CompositionalFeatureLearner(**values)


@pytest.mark.parametrize(
    "field",
    [
        "step_size_output",
        "step_size_theta",
        "promotion_margin",
        "promotion_blend",
        "obgd_kappa",
        "parent_temperature",
        "parent_novelty_weight",
        "future_utility_mix",
        "future_utility_trace_decay",
        "candidate_score_energy_epsilon",
        "candidate_selector_learning_rate",
        "generator_resource_advantage_clip",
        "generator_resource_cost_weight",
    ],
)
def test_float32_sinks_contain_hostile_ratio_and_repr_hooks(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _learner(**{field: _HostileFloat(0.5)})


@pytest.mark.parametrize(
    "field",
    [
        "step_size_output",
        "promotion_margin",
        "parent_novelty_weight",
        "future_utility_mix",
        "generator_resource_cost_weight",
    ],
)
def test_nonzero_float32_sinks_reject_underflow_to_zero(field: str) -> None:
    with pytest.raises(ValueError, match="remain nonzero"):
        _learner(**{field: 1e-100})


def test_selector_bounds_reject_nonzero_underflow() -> None:
    with pytest.raises(ValueError, match="remain nonzero"):
        FiniteCandidateSelector(2, loss_lower_bound=1e-100)


def test_selector_float_and_string_sinks_are_exact_and_canonical() -> None:
    selector = FiniteCandidateSelector(
        2,
        learning_rate=cast(Any, np.float32(0.5)),
        exploration=cast(Any, np.float64(0.25)),
    )
    assert type(selector.to_config()["learning_rate"]) is float
    for field in ("learning_rate", "exploration", "loss_lower_bound", "loss_upper_bound"):
        with pytest.raises(ValueError, match=field):
            FiniteCandidateSelector(2, **cast(Any, {field: _HostileFloat(0.5)}))
    with pytest.raises(ValueError, match="update_rule"):
        FiniteCandidateSelector(2, update_rule=np.str_("hedge"))


def test_selector_refuses_updates_after_exact_int32_horizon() -> None:
    selector = FiniteCandidateSelector(2)
    state = selector.init().replace(step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32))
    result = selector.update(state, jnp.zeros((2,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.state, state)


def test_selector_static_state_and_loss_shapes_fail_before_computation() -> None:
    selector = FiniteCandidateSelector(2)
    state = selector.init()
    with pytest.raises(TypeError, match="FiniteCandidateSelectorState"):
        selector.probabilities(object())  # type: ignore[arg-type]
    malformed = FiniteCandidateSelectorState(
        log_weights=jnp.zeros((1,), dtype=jnp.float32),
        cumulative_loss=state.cumulative_loss,
        action_counts=state.action_counts,
        step_count=state.step_count,
    )
    with pytest.raises(ValueError, match="shape or dtype"):
        selector.probabilities(malformed)
    with pytest.raises(ValueError, match="losses shape"):
        selector.update(state, jnp.zeros((1,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="losses dtype"):
        selector.update(state, jnp.zeros((2,), dtype=jnp.int32))
    with pytest.raises(ValueError, match="losses shape"):
        selector.validate_bounded_losses(jnp.zeros((1,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="losses dtype"):
        selector.validate_bounded_losses(jnp.zeros((2,), dtype=jnp.int32))


def test_exp3_selector_rejects_out_of_range_and_hostile_actions_atomically() -> None:
    selector = FiniteCandidateSelector(2, exploration=0.1, update_rule="exp3")
    state = selector.init()
    for action in (-1, 2):
        result = selector.update(
            state,
            jnp.zeros((2,), dtype=jnp.float32),
            selected_action=action,
        )
        chex.assert_trees_all_equal(result.state, state)

    class HostileInt(int):
        def __index__(self) -> int:  # pragma: no cover
            raise AssertionError("hostile action hook executed")

    with pytest.raises(TypeError, match="scalar integer"):
        selector.update(
            state,
            jnp.zeros((2,), dtype=jnp.float32),
            selected_action=HostileInt(0),
        )

    with pytest.raises(ValueError, match="selected_action is required"):
        selector.update(state, jnp.zeros((2,), dtype=jnp.float32))

    # A wide exact integer must not wrap through int32 to a valid action.
    wrapped = selector.update(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        selected_action=np.uint64(2**32),
    )
    chex.assert_trees_all_equal(wrapped.state, state)
    assert int(wrapped.selected_action) == -1


def test_exp3_invalid_array_action_does_not_index_the_untrusted_value() -> None:
    selector = FiniteCandidateSelector(2, exploration=0.1, update_rule="exp3")
    state = selector.init()
    result = selector.update(
        state,
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        selected_action=jnp.asarray(2, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(result.state, state)
    np.testing.assert_allclose(np.asarray(result.bounded_losses), np.asarray([0.25, 0.75]))

    update_jit = jax.jit(
        lambda current, action: selector.update(
            current,
            jnp.asarray([0.25, 0.75], dtype=jnp.float32),
            selected_action=action,
        )
    )
    valid = update_jit(state, jnp.asarray(1, dtype=jnp.int32))
    assert int(valid.state.step_count) == 1
    rejected = update_jit(state, jnp.asarray(2, dtype=jnp.int32))
    chex.assert_trees_all_equal(rejected.state, state)


def test_selector_rejects_semantically_corrupt_state_atomically() -> None:
    selector = FiniteCandidateSelector(2)
    state = selector.init()
    losses = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    corruptions = (
        state.replace(cumulative_loss=state.cumulative_loss.at[0].set(-1.0)),
        state.replace(action_counts=state.action_counts.at[0].set(-1.0)),
        state.replace(action_counts=state.action_counts.at[0].set(1.0)),
        state.replace(cumulative_loss=state.cumulative_loss.at[0].set(1.0)),
    )
    for corrupt in corruptions:
        result = selector.update(corrupt, losses)
        chex.assert_trees_all_equal(result.state, corrupt)


@pytest.mark.parametrize("horizon", [True, 1.5, np.float32(1.0), 0, 2**31])
def test_selector_regret_horizon_is_an_exact_bounded_integer(horizon: object) -> None:
    selector = FiniteCandidateSelector(2)
    with pytest.raises(ValueError, match="horizon"):
        selector.regret_metadata(cast(Any, horizon))

    metadata = selector.regret_metadata(np.int64(2**31 - 1))
    assert metadata["horizon"] == 2**31 - 1
    assert type(metadata["horizon"]) is int


def test_selector_regret_metadata_accounts_for_exploration_and_exp3_tuning() -> None:
    horizon = 20
    base = FiniteCandidateSelector(2, learning_rate=0.5).regret_metadata(horizon)
    explored = FiniteCandidateSelector(
        2,
        learning_rate=0.5,
        exploration=0.25,
    ).regret_metadata(horizon)
    assert explored["regret_bound"] == pytest.approx(base["regret_bound"] + 0.25 * horizon)

    exp3 = FiniteCandidateSelector(
        2,
        learning_rate=0.5,
        exploration=0.25,
        update_rule="exp3",
    ).regret_metadata(horizon)
    assert exp3["regret_bound"] == pytest.approx(float(horizon))
    assert exp3["bound_kind"] == "worst_case"


def test_compositional_update_refuses_exhausted_or_corrupt_integer_state() -> None:
    learner = _learner()
    state = learner.init(feature_dim=2, key=jr.key(0))
    exhausted = state.replace(
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        ages=jnp.full_like(state.ages, 2**31 - 1),
        candidate_ages=jnp.full_like(state.candidate_ages, 2**31 - 1),
    )
    event = (
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    rejected = learner.update(exhausted, *event)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(
        rejected.state,
        exhausted.replace(
            birth_timestamp=rejected.state.birth_timestamp,
            uptime_s=rejected.state.uptime_s,
        ),
    )

    corrupt = state.replace(ages=state.ages.at[0].set(-1))
    rejected = learner.update(corrupt, *event)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(
        rejected.state,
        corrupt.replace(
            birth_timestamp=rejected.state.birth_timestamp,
            uptime_s=rejected.state.uptime_s,
        ),
    )


def test_sequence_and_string_contracts_roundtrip_canonically() -> None:
    learner = _learner(
        operation_prior=(0.0, 1.0, 0.0, 0.0, 0.0),
        generator_resource_initial_preferences=(0.0, 0.0, 0.0, 0.0),
    )
    assert CompositionalFeatureLearner.from_config(learner.to_config()).to_config() == (
        learner.to_config()
    )
    with pytest.raises(ValueError, match="operation_prior"):
        _learner(operation_prior=[0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="generation_strategy"):
        _learner(generation_strategy=np.str_("utility"))


def test_signed_scaffolds_preflight_only_when_they_can_be_constructed() -> None:
    learner = _learner(
        n_features=20_001,
        n_tasks=1,
        candidate_count=0,
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        signed_tanh_scaffold_count=1,
    )
    with pytest.raises(ValueError, match="signed-tanh scaffold construction"):
        learner.init(20_000, jr.key(0))

    # With no composed slot, init never constructs the quadratic parent tables.
    no_composition = _learner(
        n_features=10,
        n_tasks=1,
        candidate_count=0,
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        signed_tanh_scaffold_count=1,
    )
    assert no_composition.init(10, jr.key(1)).ops.shape == (10,)
