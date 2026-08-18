"""Mechanism tests for conventional option value plus expected duration.

These are analytic and deterministic development tests, not held-out evidence
that Alberta Plan Step 5 is complete.
"""

import json
import math
from collections.abc import Iterator, Mapping
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.option_value_duration import (
    DURATION_HEAD,
    OptionValueDurationConfig,
    OptionValueDurationLearner,
    run_option_value_duration_from_arrays,
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


@pytest.mark.parametrize(
    "integer_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.longlong,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.ulonglong,
    ],
)
def test_option_duration_canonicalizes_numpy_integer_family(integer_type) -> None:
    learner = OptionValueDurationLearner(integer_type(2))

    assert type(learner.n_options) is int
    assert learner.trainable_parameter_count(integer_type(3)) == 12
    assert learner.init(integer_type(3)).weights.shape == (2, 2, 3)


@pytest.mark.parametrize("field", ["n_options", "feature_dim"])
def test_option_duration_rejects_hostile_integer_subclasses(field: str) -> None:
    class HostileInt(int):
        def __index__(self) -> int:
            raise AssertionError("untrusted index hook executed")

        def __repr__(self) -> str:
            raise AssertionError("untrusted repr hook executed")

    if field == "n_options":
        with pytest.raises(ValueError, match=field):
            OptionValueDurationLearner(HostileInt(2))
    else:
        learner = OptionValueDurationLearner(2)
        with pytest.raises(ValueError, match=field):
            learner.init(HostileInt(3))


def test_option_duration_float_subclasses_are_rejected_without_hooks() -> None:
    class CountingFloat(float):
        def __new__(cls):
            instance = super().__new__(cls, 0.25)
            instance.calls = 0
            return instance

        def as_integer_ratio(self) -> tuple[int, int]:
            self.calls += 1
            return (1, 4)

    value = CountingFloat()
    with pytest.raises(ValueError, match="reward_step_size"):
        OptionValueDurationConfig(reward_step_size=value)
    assert value.calls == 0

    class ExplodingFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            raise RuntimeError("hostile ratio")

        def __repr__(self) -> str:
            raise AssertionError("untrusted repr hook executed")

    with pytest.raises(ValueError, match="reward_step_size"):
        OptionValueDurationConfig(reward_step_size=ExplodingFloat(0.25))


def test_option_duration_rejects_builtin_float32_underflow() -> None:
    with pytest.raises(ValueError, match="reward_step_size"):
        OptionValueDurationConfig(reward_step_size=1.0e-50)


def test_option_duration_config_preserves_historical_loader_forms() -> None:
    class ConfigSubclass(OptionValueDurationConfig):
        pass

    with pytest.raises(ValueError, match="actual OptionValueDurationConfig"):
        OptionValueDurationLearner(2, ConfigSubclass())

    learner_payload = OptionValueDurationLearner(2).to_config()
    learner_payload["n_options"] = np.int32(2)
    restored = OptionValueDurationLearner.from_config(MappingProxyType(learner_payload))
    assert restored.n_options == 2

    config_payload = {"type": "historical-marker", "reward_step_size": np.float32(0.25)}
    config = OptionValueDurationConfig.from_config(MappingProxyType(config_payload))
    assert config.reward_step_size == 0.25
    assert config.duration_step_size == 0.1

    learner_payload["type"] = "historical-learner-marker"
    assert OptionValueDurationLearner.from_config(learner_payload).n_options == 2


def test_option_duration_loaders_normalize_hostile_mapping_failures_without_repr() -> None:
    class HostileMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("hostile mapping")

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("hostile mapping")

        def __len__(self) -> int:
            raise RuntimeError("hostile mapping")

        def __repr__(self) -> str:
            raise AssertionError("untrusted repr hook executed")

    with pytest.raises(ValueError, match="readable mapping"):
        OptionValueDurationConfig.from_config(HostileMapping())
    with pytest.raises(ValueError, match="readable mapping"):
        OptionValueDurationLearner.from_config(HostileMapping())


def test_option_duration_resource_formula_matches_materialized_state() -> None:
    learner = OptionValueDurationLearner(3)
    state = learner.init(5)
    actual_bytes = sum(
        int(leaf.nbytes)
        for leaf in jax.tree_util.tree_leaves(state)
        if hasattr(leaf, "nbytes")
    )

    assert learner.persistent_resource_budget(5) == {
        "trainable_parameters": 30,
        "persistent_scalars": 34,
        "persistent_bytes": actual_bytes,
    }


def test_option_duration_resource_limit_precedes_allocation(monkeypatch) -> None:
    learner = OptionValueDurationLearner(1)
    last_valid = ((256 * 1024 * 1024 // 4) - 2) // 2
    assert learner.persistent_resource_budget(last_valid)["persistent_bytes"] <= 256 * 1024 * 1024

    monkeypatch.setattr(
        jnp,
        "zeros",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("allocation began before resource preflight")
        ),
    )
    with pytest.raises(ValueError, match="256 MiB"):
        learner.init(last_valid + 1)

    last_valid_options = ((256 * 1024 * 1024 // 4) - 1) // 3
    OptionValueDurationLearner(last_valid_options)
    with pytest.raises(ValueError, match="256 MiB"):
        OptionValueDurationLearner(last_valid_options + 1)


def test_option_duration_runtime_shape_dtype_and_state_contracts() -> None:
    learner = OptionValueDurationLearner(2)
    state = learner.init(3)
    malformed = state.replace(weights=jnp.zeros((2, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="state.weights"):
        learner.predict(malformed, jnp.ones(3, dtype=jnp.float32))

    with pytest.raises(ValueError, match="observation"):
        learner.predict(state, jnp.ones((1, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="dtype"):
        learner.predict(state, jnp.ones(3, dtype=jnp.int32))
    with pytest.raises(ValueError, match="option_index"):
        learner.update(
            state,
            jnp.ones(3, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.ones(3, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )

    class HostileArray:
        @property
        def ndim(self):
            raise RuntimeError("hostile shape")

        def __repr__(self) -> str:
            raise AssertionError("untrusted repr hook executed")

    hostile_state = state.replace(weights=HostileArray())
    with pytest.raises(ValueError, match="readable shape and dtype"):
        learner._require_state_contract(hostile_state)


def test_option_duration_invalid_dynamic_state_is_held_atomically() -> None:
    learner = OptionValueDurationLearner(1)
    state = learner.init(1).replace(
        option_update_counts=jnp.asarray([-1], dtype=jnp.int32)
    )
    result = learner.update(
        state,
        jnp.ones(1, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(result.state.weights, state.weights)
    chex.assert_trees_all_equal(
        result.state.option_update_counts, state.option_update_counts
    )
    chex.assert_trees_all_equal(result.state.step_count, state.step_count)
    assert not bool(result.update_applied)


def test_option_duration_array_runner_requires_exact_shapes() -> None:
    learner = OptionValueDurationLearner(1)
    state = learner.init(2)
    with pytest.raises(ValueError, match="option_indices"):
        run_option_value_duration_from_arrays(
            learner,
            state,
            jnp.ones((2, 2), dtype=jnp.float32),
            jnp.zeros((2, 1), dtype=jnp.int32),
            jnp.ones((2,), dtype=jnp.float32),
            jnp.ones((2, 2), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
        )


def test_array_runner_rejects_untrusted_inputs_without_running_conversion_hooks() -> None:
    """The array runner must gate on array metadata before any conversion.

    ``_require_array`` reached ``jnp.asarray(value)`` before validating
    anything, so an untrusted ``__array__`` hook executed ahead of the shape
    and dtype checks and chose the value they then saw. The public ``update``
    path is gated by an outer contract check; the array runner called the
    helper directly.
    """

    class _HostileArray:
        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            raise AssertionError("array hook must not run")

    learner = OptionValueDurationLearner(3, OptionValueDurationConfig())
    state = learner.init(2)
    steps = 2
    observations = jnp.zeros((steps, 2), dtype=jnp.float32)
    next_observations = jnp.zeros((steps, 2), dtype=jnp.float32)
    rewards = jnp.zeros((steps,), dtype=jnp.float32)
    discounts = jnp.zeros((steps,), dtype=jnp.float32)
    option_indices = jnp.zeros((steps,), dtype=jnp.int32)

    with pytest.raises(TypeError, match="shape and dtype metadata"):
        run_option_value_duration_from_arrays(
            learner,
            state,
            observations,
            _HostileArray(),
            rewards,
            next_observations,
            discounts,
        )

    with pytest.raises(TypeError, match="shape and dtype metadata"):
        run_option_value_duration_from_arrays(
            learner,
            state,
            observations,
            option_indices,
            _HostileArray(),
            next_observations,
            discounts,
        )


def test_array_runner_still_accepts_trusted_arrays() -> None:
    """The metadata gate must not reject the arrays the runner already supports."""
    learner = OptionValueDurationLearner(3, OptionValueDurationConfig())
    state = learner.init(2)
    steps = 2
    result = run_option_value_duration_from_arrays(
        learner,
        state,
        jnp.zeros((steps, 2), dtype=jnp.float32),
        np.zeros((steps,), dtype=np.int32),
        jnp.zeros((steps,), dtype=jnp.float32),
        jnp.zeros((steps, 2), dtype=jnp.float32),
        jnp.zeros((steps,), dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(result.predictions)
