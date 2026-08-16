"""Mechanism tests for conventional option value plus expected duration.

These are analytic and deterministic development tests, not held-out evidence
that Alberta Plan Step 5 is complete.
"""

import json
import math
from decimal import Decimal
from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.option_value_duration import (
    DURATION_HEAD,
    OptionValueDurationConfig,
    OptionValueDurationLearner,
)

pytestmark = pytest.mark.unit


def test_config_roundtrip_validation_and_fixed_parameter_count() -> None:
    config = OptionValueDurationConfig(
        reward_step_size=0.2,
        duration_step_size=0.3,
        duration_floor=1e-4,
    )
    learner = OptionValueDurationLearner.from_config(
        OptionValueDurationLearner(3, config).to_config()
    )

    assert learner.n_options == 3
    assert learner.config == config
    assert learner.trainable_parameter_count(feature_dim=5) == 3 * 2 * 5
    chex.assert_shape(learner.init(5).weights, (3, 2, 5))

    with pytest.raises(ValueError, match="n_options"):
        OptionValueDurationLearner(0)
    with pytest.raises(ValueError, match="reward_step_size"):
        OptionValueDurationConfig(reward_step_size=-0.1)
    with pytest.raises(ValueError, match="duration_step_size"):
        OptionValueDurationConfig(duration_step_size=-0.1)
    with pytest.raises(ValueError, match="duration_floor"):
        OptionValueDurationConfig(duration_floor=0.0)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="reward_step_size"):
            OptionValueDurationConfig(reward_step_size=invalid)
        with pytest.raises(ValueError, match="duration_step_size"):
            OptionValueDurationConfig(duration_step_size=invalid)
        with pytest.raises(ValueError, match="duration_floor"):
            OptionValueDurationConfig(duration_floor=invalid)


@pytest.mark.parametrize(
    "invalid",
    (True, np.bool_(True), 1.5, np.float64(2.0), "2", None, -1, 2_147_483_648),
)
def test_learner_rejects_noncanonical_option_counts(invalid: object) -> None:
    with pytest.raises(ValueError, match="n_options"):
        OptionValueDurationLearner(invalid)  # type: ignore[arg-type]

    payload = OptionValueDurationLearner(2).to_config()
    payload["n_options"] = invalid
    with pytest.raises(ValueError, match="n_options"):
        OptionValueDurationLearner.from_config(payload)


@pytest.mark.parametrize(
    "invalid",
    (True, np.bool_(True), 1.5, np.float64(2.0), "2", None, -1),
)
def test_learner_rejects_noncanonical_feature_dimensions(invalid: object) -> None:
    learner = OptionValueDurationLearner(2)
    with pytest.raises(ValueError, match="feature_dim"):
        learner.trainable_parameter_count(invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        learner.init(invalid)  # type: ignore[arg-type]


def test_learner_rejects_feature_dimensions_above_int32() -> None:
    learner = OptionValueDurationLearner(2)
    with pytest.raises(ValueError, match="feature_dim"):
        learner.trainable_parameter_count(2_147_483_648)


@pytest.mark.parametrize(
    "field",
    ("reward_step_size", "duration_step_size", "duration_floor"),
)
@pytest.mark.parametrize(
    "invalid",
    (
        True,
        np.bool_(True),
        "0.1",
        1.0 + 0.0j,
        Decimal("0.1"),
        jnp.asarray(0.1, dtype=jnp.float32),
    ),
)
def test_config_rejects_boolean_and_non_real_scalar_inputs(
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        OptionValueDurationConfig(**{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("reward_step_size", Fraction(-1, 10**400)),
        ("duration_step_size", Fraction(-1, 10**400)),
        ("duration_floor", Fraction(-1, 10**400)),
        ("reward_step_size", Fraction(1, 10**400)),
        ("duration_step_size", Fraction(1, 10**400)),
        ("duration_floor", Fraction(1, 10**400)),
        ("reward_step_size", 3.5e38),
        ("duration_step_size", 3.5e38),
        ("duration_floor", 3.5e38),
    ),
)
def test_config_enforces_exact_domain_and_float32_sink_bounds(
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        OptionValueDurationConfig(**{field: invalid})


def test_config_uses_direct_float32_narrowing_without_double_rounding() -> None:
    overflow_midpoint = np.ldexp(
        np.longdouble(2) - np.ldexp(np.longdouble(1), -24),
        127,
    )
    largest_finite_input = np.nextafter(
        overflow_midpoint,
        np.longdouble("-inf"),
    )

    reward = OptionValueDurationConfig(reward_step_size=largest_finite_input)
    duration = OptionValueDurationConfig(duration_step_size=largest_finite_input)
    floor = OptionValueDurationConfig(duration_floor=largest_finite_input)

    for value in (
        reward.reward_step_size,
        duration.duration_step_size,
        floor.duration_floor,
    ):
        assert type(value) is float
        assert bool(np.isfinite(np.asarray(value, dtype=np.float32)))


@pytest.mark.parametrize(
    "field",
    ("reward_step_size", "duration_step_size", "duration_floor"),
)
def test_config_rounds_exact_rationals_to_float32_with_ties_to_even(field: str) -> None:
    midpoint = Fraction(1, 1) + Fraction(1, 2**24)
    offset = Fraction(1, 2**60)
    next_float32 = np.nextafter(
        np.float32(1.0),
        np.float32(2.0),
        dtype=np.float32,
    )

    below = OptionValueDurationConfig(**{field: midpoint - offset})
    tie = OptionValueDurationConfig(**{field: midpoint})
    above = OptionValueDurationConfig(**{field: midpoint + offset})

    assert getattr(below, field) == 1.0
    assert getattr(tie, field) == 1.0
    assert getattr(above, field) == float(next_float32)


@pytest.mark.parametrize(
    "field",
    ("reward_step_size", "duration_step_size", "duration_floor"),
)
def test_config_enforces_exact_float32_underflow_and_overflow_midpoints(field: str) -> None:
    minimum_subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    underflow_midpoint = Fraction(1, 2**150)
    overflow_midpoint = Fraction(((2**24 - 1) * 2**104) + 2**103)

    with pytest.raises(ValueError, match=field):
        OptionValueDurationConfig(**{field: underflow_midpoint})
    above_underflow = OptionValueDurationConfig(
        **{field: underflow_midpoint + Fraction(1, 2**200)}
    )
    assert getattr(above_underflow, field) == minimum_subnormal

    below_overflow = OptionValueDurationConfig(**{field: overflow_midpoint - 1})
    assert getattr(below_overflow, field) == float(np.finfo(np.float32).max)
    with pytest.raises(ValueError, match=field):
        OptionValueDurationConfig(**{field: overflow_midpoint})


def test_config_canonicalizes_real_scalars_and_preserves_builtin_float_payload() -> None:
    builtin = OptionValueDurationConfig(
        reward_step_size=0.1,
        duration_step_size=0.2,
        duration_floor=1.0e-4,
    )
    assert builtin.to_config() == {
        "type": "OptionValueDurationConfig",
        "reward_step_size": 0.1,
        "duration_step_size": 0.2,
        "duration_floor": 1.0e-4,
    }

    config = OptionValueDurationConfig(
        reward_step_size=np.float64(0.25),
        duration_step_size=np.int64(1),
        duration_floor=Fraction(1, 4),
    )
    payload = config.to_config()

    assert type(config.reward_step_size) is float
    assert type(config.duration_step_size) is float
    assert type(config.duration_floor) is float
    json.dumps(payload, allow_nan=False)
    assert OptionValueDurationConfig.from_config(payload) == config

    signed_zero = OptionValueDurationConfig(
        reward_step_size=-0.0,
        duration_step_size=-0.0,
    )
    assert math.copysign(1.0, signed_zero.reward_step_size) < 0.0
    assert math.copysign(1.0, signed_zero.duration_step_size) < 0.0


def test_two_head_td_targets_and_updates_match_exact_analytic_values() -> None:
    learner = OptionValueDurationLearner(
        2,
        OptionValueDurationConfig(
            reward_step_size=0.1,
            duration_step_size=0.2,
        ),
    )
    initial_weights = jnp.array(
        [
            [[2.0, -1.0], [0.5, 1.5]],
            [[7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    state = learner.init(2).replace(weights=initial_weights)  # type: ignore[attr-defined]

    result = jax.jit(learner.update)(
        state,
        jnp.array([1.0, 2.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(3.0, dtype=jnp.float32),
        jnp.array([2.0, -1.0], dtype=jnp.float32),
        jnp.array(0.75, dtype=jnp.float32),
    )

    # predictions = [0, 3.5], next_predictions = [5, -0.5].
    # targets = [3, 1] + 0.75 * next_predictions = [6.75, 0.625].
    chex.assert_trees_all_close(
        result.predictions,
        jnp.array([0.0, 3.5], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.next_predictions,
        jnp.array([5.0, -0.5], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.td_targets,
        jnp.array([6.75, 0.625], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.td_errors,
        jnp.array([6.75, -2.875], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.state.weights[0],
        jnp.array([[2.675, 0.35], [-0.075, 0.35]], dtype=jnp.float32),
        atol=1e-6,
    )
    chex.assert_trees_all_close(result.state.weights[1], initial_weights[1])
    chex.assert_trees_all_equal(
        result.state.option_update_counts,
        jnp.array([1, 0], dtype=jnp.int32),
    )
    assert int(result.state.step_count) == 1


def test_applied_update_saturates_lifetime_counters_without_wrapping() -> None:
    learner = OptionValueDurationLearner(1)
    int32_max = jnp.iinfo(jnp.int32).max
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        option_update_counts=jnp.array([int32_max], dtype=jnp.int32),
        step_count=jnp.array(int32_max, dtype=jnp.int32),
    )

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert not bool(jnp.array_equal(result.state.weights, state.weights))
    assert int(result.state.option_update_counts[0]) == int32_max
    assert int(result.state.step_count) == int32_max


def test_termination_discount_zeros_bootstrap_and_no_average_reward_is_subtracted() -> None:
    learner = OptionValueDurationLearner(
        1,
        OptionValueDurationConfig(
            reward_step_size=0.0,
            duration_step_size=0.0,
        ),
    )
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        weights=jnp.array([[[2.0], [7.0]]], dtype=jnp.float32)
    )

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(5.0, dtype=jnp.float32),
        jnp.array([100.0], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    # An arbitrarily large next prediction cannot leak through termination.
    # The conventional reward target is the raw reward, not reward minus rbar.
    chex.assert_trees_all_close(
        result.td_targets,
        jnp.array([5.0, 1.0], dtype=jnp.float32),
    )
    assert not hasattr(result.state, "average_reward")


def test_termination_does_not_multiply_inf_next_prediction() -> None:
    """gamma=0 * inf V(s') is 0*inf = NaN and would freeze both option heads."""
    learner = OptionValueDurationLearner(
        1,
        OptionValueDurationConfig(
            reward_step_size=0.1,
            duration_step_size=0.0,
        ),
    )
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        weights=jnp.array([[[2.0], [7.0]]], dtype=jnp.float32)
    )
    next_obs = jnp.array([jnp.inf], dtype=jnp.float32)
    raw = jnp.asarray(0.0, dtype=jnp.float32) * (jnp.array([2.0], dtype=jnp.float32) @ next_obs)
    assert not bool(jnp.isfinite(raw))

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(5.0, dtype=jnp.float32),
        next_obs,
        jnp.array(0.0, dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        result.td_targets,
        jnp.array([5.0, 1.0], dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(result.state.weights)
    chex.assert_tree_all_finite(result.next_predictions)
    assert bool(result.update_applied)


def test_reward_rate_prediction_preserves_raw_duration_and_floors_only_score() -> None:
    learner = OptionValueDurationLearner(
        2,
        OptionValueDurationConfig(duration_floor=0.5),
    )
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        weights=jnp.array(
            [
                [[6.0], [10.0]],
                [[4.0], [0.0]],
            ],
            dtype=jnp.float32,
        )
    )

    prediction = learner.predict(state, jnp.array([1.0], dtype=jnp.float32))

    chex.assert_trees_all_close(
        prediction.reward_values,
        jnp.array([6.0, 4.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        prediction.durations,
        jnp.array([10.0, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        prediction.reward_rates,
        jnp.array([0.6, 8.0], dtype=jnp.float32),
    )


def test_infinite_reward_on_zero_feature_does_not_poison_duration_head() -> None:
    """Inf reward is 0*inf = NaN on a silent feature of the reward head.

    The duration head's TD error stays finite. Map NaN products back to the
    previous weight so that head can keep learning, and leave genuine infs.
    """
    learner = OptionValueDurationLearner(
        1,
        OptionValueDurationConfig(reward_step_size=0.1, duration_step_size=0.2),
    )
    state = learner.init(2)
    obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    nxt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    option = jnp.array(0, dtype=jnp.int32)
    discount = jnp.array(0.0, dtype=jnp.float32)

    poisoned = learner.update(
        state, obs, option, jnp.array(jnp.inf, dtype=jnp.float32), nxt, discount
    )
    assert not bool(jnp.any(jnp.isnan(poisoned.state.weights)))
    chex.assert_tree_all_finite(poisoned.state.weights[0, DURATION_HEAD])
    chex.assert_trees_all_close(
        poisoned.state.weights[0, 0, 0],
        state.weights[0, 0, 0],
    )
    assert bool(poisoned.update_applied)
    chex.assert_trees_all_equal(
        poisoned.head_updates_applied,
        jnp.array([False, True]),
    )
    assert float(poisoned.td_errors[0]) == 0.0
    assert bool(jnp.isfinite(poisoned.td_errors[DURATION_HEAD]))

    recovered = learner.update(
        poisoned.state, obs, option, jnp.array(1.0, dtype=jnp.float32), nxt, discount
    )
    assert not bool(jnp.any(jnp.isnan(recovered.state.weights)))
    chex.assert_tree_all_finite(recovered.state.weights[0, DURATION_HEAD])
    assert bool(recovered.update_applied)
    assert bool(jnp.all(recovered.head_updates_applied))
