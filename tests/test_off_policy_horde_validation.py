"""Complete scalar, resource, state, and input contracts for shared GTD Horde."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.off_policy_horde import (
    NonlinearSharedGTDHordeLearner,
)
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec, create_horde_spec

_INT32_MAX = 2**31 - 1


def _spec(count: int = 1) -> HordeSpec:
    return create_horde_spec(
        tuple(
            GVFSpec(
                name=f"demon-{index}",
                demon_type=DemonType.PREDICTION,
                gamma=0.8,
                lamda=0.0,
                cumulant_index=index,
            )
            for index in range(count)
        )
    )


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
def test_shared_gtd_accepts_full_numpy_integer_family(integer_type: type) -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(), hidden_size=integer_type(2))
    assert learner.init(integer_type(2), jax.random.key(0)).trunk_w.shape == (2, 2)


def test_shared_gtd_rejects_hostile_integer_subclass_without_hooks() -> None:
    class HostileInt(int):
        def __index__(self) -> int:  # pragma: no cover
            raise AssertionError("conversion hook executed")

        def __repr__(self) -> str:  # pragma: no cover
            raise AssertionError("repr hook executed")

    with pytest.raises(ValueError, match="hidden_size"):
        NonlinearSharedGTDHordeLearner(_spec(), hidden_size=HostileInt(2))
    learner = NonlinearSharedGTDHordeLearner(_spec())
    with pytest.raises(ValueError, match="feature_dim"):
        learner.init(HostileInt(2), jax.random.key(0))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("primary_step_size", True),
        ("primary_step_size", 1e100),
        ("primary_step_size", 1e-100),
        ("secondary_step_size", 0.0),
        ("ratio_clip", float("nan")),
        ("init_scale", float("inf")),
    ],
)
def test_shared_gtd_float32_sinks_fail_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        NonlinearSharedGTDHordeLearner(_spec(), **{field: value})


def test_shared_gtd_state_product_is_preflighted_without_allocation() -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(2), hidden_size=200_000)
    with pytest.raises(ValueError, match="persistent state bytes"):
        learner.init(1_000, jax.random.key(0))


def test_shared_gtd_rejects_malformed_horde_spec_metadata() -> None:
    spec = _spec(2)
    with pytest.raises(ValueError, match="gammas must have shape"):
        NonlinearSharedGTDHordeLearner(
            spec.replace(gammas=jnp.zeros((1,), dtype=jnp.float32))
        )
    with pytest.raises(TypeError, match="lamdas must have dtype float32"):
        NonlinearSharedGTDHordeLearner(
            spec.replace(lamdas=jnp.zeros((2,), dtype=jnp.int32))
        )


def test_shared_gtd_validates_adopted_state_shapes_and_dtypes() -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(2), hidden_size=3)
    state = learner.init(2, jax.random.key(0))
    obs = jnp.ones((2,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="secondary_trunk_w has an invalid shape"):
        learner.predict(
            state.replace(
                secondary_trunk_w=jnp.zeros((2, 3, 3), dtype=jnp.float32)
            ),
            obs,
        )
    with pytest.raises(TypeError, match="head_w has an invalid dtype"):
        learner.predict(
            state.replace(head_w=jnp.zeros((2, 3), dtype=jnp.int32)),
            obs,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("observation", jnp.ones((3,), dtype=jnp.float32), "observation must have shape"),
        ("cumulants", jnp.ones((2,), dtype=jnp.float32), "cumulants must have shape"),
        ("rhos", jnp.ones((1,), dtype=jnp.int32), "rhos must have dtype"),
        ("discounts", jnp.ones((2,), dtype=jnp.float32), "discounts must have shape"),
    ],
)
def test_shared_gtd_public_input_metadata_fails_before_tracing(
    field: str, replacement: jax.Array, match: str
) -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(), hidden_size=3)
    state = learner.init(2, jax.random.key(0))
    values = {
        "observation": jnp.ones((2,), dtype=jnp.float32),
        "cumulants": jnp.ones((1,), dtype=jnp.float32),
        "next_observation": jnp.ones((2,), dtype=jnp.float32),
        "rhos": jnp.ones((1,), dtype=jnp.float32),
        "discounts": jnp.full((1,), 0.8, dtype=jnp.float32),
    }
    values[field] = replacement
    with pytest.raises((TypeError, ValueError), match=match):
        learner.update_with_ratios_and_discounts(state, **values)


@pytest.mark.parametrize(
    ("rhos", "discounts"),
    [([-1.0], [0.8]), ([1.0], [-0.1]), ([1.0], [1.1])],
)
def test_shared_gtd_rejects_invalid_ratio_and_discount_domains(
    rhos: list[float], discounts: list[float]
) -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(), hidden_size=3)
    state = learner.init(2, jax.random.key(0))
    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        jnp.asarray(rhos, dtype=jnp.float32),
        jnp.asarray(discounts, dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_shared_gtd_terminal_transition_ignores_nonfinite_next_observation() -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(), hidden_size=3)
    state = learner.init(2, jax.random.key(0))
    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    chex.assert_tree_all_finite(result.state)


def test_shared_gtd_step_counter_saturates_and_negative_counter_rolls_back() -> None:
    learner = NonlinearSharedGTDHordeLearner(_spec(), hidden_size=3)
    initial = learner.init(2, jax.random.key(0))
    args = (
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.full((1,), 0.8, dtype=jnp.float32),
    )
    terminal = initial.replace(step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32))
    saturated = learner.update_with_ratios_and_discounts(terminal, *args)
    assert bool(saturated.update_applied)
    assert int(saturated.state.step_count) == _INT32_MAX

    invalid = initial.replace(step_count=jnp.asarray(-1, dtype=jnp.int32))
    rejected = learner.update_with_ratios_and_discounts(invalid, *args)
    assert not bool(rejected.update_applied)
    assert int(rejected.state.step_count) == -1
