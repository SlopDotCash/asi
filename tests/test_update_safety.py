"""Tests for fail-closed host identities on discrete-action safety."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.update_safety import safe_discrete_action


def test_legal_action_in_domain_is_valid() -> None:
    safe, valid = safe_discrete_action(1, 3, allow_unset=False)
    assert int(safe) == 1
    assert bool(valid)


def test_legal_unset_sentinel_is_valid() -> None:
    safe, valid = safe_discrete_action(-1, 3, allow_unset=True)
    assert int(safe) == -1
    assert bool(valid)


@pytest.mark.parametrize("shape", [(1,), (1, 1)])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32])
def test_safe_discrete_action_rejects_nonscalar_aliases(
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> None:
    action = jnp.ones(shape, dtype=dtype)

    with pytest.raises(ValueError, match="action must be a scalar"):
        safe_discrete_action(action, 3)
    with pytest.raises(ValueError, match="action must be a scalar"):
        safe_discrete_action(np.asarray(action), 3)
    with pytest.raises(ValueError, match="action must be a scalar"):
        jax.jit(lambda value: safe_discrete_action(value, 3))(action)


def test_integer_zero_actions_keeps_empty_domain_branch() -> None:
    safe, valid = safe_discrete_action(0, 0, allow_unset=False)
    assert int(safe) == 0
    assert bool(valid)


def test_out_of_range_action_is_invalid() -> None:
    safe, valid = safe_discrete_action(3, 3, allow_unset=False)
    assert int(safe) == 0
    assert not bool(valid)


@pytest.mark.parametrize(
    "n_actions",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param(1.5, id="float-alias"),
        pytest.param("2", id="string-two"),
        pytest.param(np.bool_(True), id="numpy-bool-true"),
    ],
)
def test_n_actions_rejects_non_int_identities(n_actions) -> None:
    with pytest.raises(ValueError, match="n_actions"):
        safe_discrete_action(0, n_actions, allow_unset=False)


@pytest.mark.parametrize(
    "allow_unset",
    [
        pytest.param(1, id="int-one"),
        pytest.param(0, id="int-zero"),
        pytest.param("yes", id="string-yes"),
        pytest.param(np.bool_(True), id="numpy-bool-true"),
        pytest.param(np.bool_(False), id="numpy-bool-false"),
    ],
)
def test_allow_unset_rejects_non_bool_identities(allow_unset) -> None:
    with pytest.raises(ValueError, match="allow_unset"):
        safe_discrete_action(0, 2, allow_unset=allow_unset)


def test_numpy_int_n_actions_remains_legal() -> None:
    safe, valid = safe_discrete_action(1, np.int32(3), allow_unset=False)
    assert int(safe) == 1
    assert bool(valid)
    assert jnp.issubdtype(safe.dtype, jnp.integer)


@pytest.mark.parametrize(
    "n_actions",
    [
        pytest.param(-1, id="negative"),
        pytest.param(2**31, id="builtin-above-int32"),
        pytest.param(np.uint64(2**32), id="numpy-above-int32"),
    ],
)
def test_n_actions_must_fit_the_int32_action_sink(n_actions: object) -> None:
    with pytest.raises(ValueError, match=r"n_actions.*\[0, 2147483647\]"):
        safe_discrete_action(0, n_actions)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "action",
    [
        2**24 + 1,
        np.int32(2**24 + 1),
        jnp.asarray(2**24 + 1, dtype=jnp.int32),
        jnp.asarray(2**24 + 1, dtype=jnp.uint32),
    ],
)
def test_integer_action_above_float32_mantissa_keeps_its_identity(action: object) -> None:
    # float32 rounds 2**24 + 1 to 2**24; an exact integer code must not be
    # silently remapped onto a neighbouring action/context row.
    safe, valid = safe_discrete_action(action, 2**24 + 2)  # type: ignore[arg-type]
    assert bool(valid)
    assert int(safe) == 2**24 + 1


def test_wide_unsigned_integer_action_is_rejected_not_wrapped() -> None:
    safe, valid = safe_discrete_action(jnp.asarray(2**32 - 1, dtype=jnp.uint32), 3)
    assert not bool(valid)
    assert int(safe) == 0
    safe, valid = safe_discrete_action(2**40, 3)
    assert not bool(valid)
    assert int(safe) == 0


@pytest.mark.parametrize("dtype, n_actions", [
    (jnp.int8, 300), (jnp.uint8, 300), (jnp.int16, 70000), (jnp.uint16, 70000),
])
def test_checked_actions_narrow_dtype_keeps_larger_domain(dtype, n_actions) -> None:
    from alberta_framework.core.update_safety import checked_integer_action_array

    expected = [0, 1, int(np.iinfo(dtype).max)]
    actions = jnp.asarray(expected, dtype=dtype)

    def check(values):
        return checked_integer_action_array(
            values, n_actions, name="actions", expected_shape=(3,), range_message="bad action",
        )

    for safe, valid in (check(actions), jax.jit(check)(actions)):
        np.testing.assert_array_equal(safe, expected)
        np.testing.assert_array_equal(valid, [True, True, True])
        assert safe.dtype == jnp.int32


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.uint8, jnp.int16, jnp.uint16])
def test_checked_actions_narrow_invalid_values_stay_rejected(dtype) -> None:
    from alberta_framework.core.update_safety import checked_integer_action_array

    actions = jnp.asarray([0, 1, 2], dtype=dtype)

    def check(values):
        return checked_integer_action_array(
            values, 2, name="actions", expected_shape=(3,), range_message="bad action",
        )

    with pytest.raises(ValueError, match="bad action"):
        check(actions)
    safe, valid = jax.jit(check)(actions)
    np.testing.assert_array_equal(safe, [0, 1, 0])
    np.testing.assert_array_equal(valid, [True, True, False])


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.uint8, jnp.int16, jnp.uint16])
def test_actor_critic_narrow_actions_match_int32_scan(dtype) -> None:
    from alberta_framework.core.actor_critic import (
        ActorCriticAgent,
        ActorCriticConfig,
        run_actor_critic_from_arrays,
    )

    agent = ActorCriticAgent(ActorCriticConfig(n_actions=300))
    state = agent.init(feature_dim=1, key=jax.random.key(0))

    def run(actions):
        return run_actor_critic_from_arrays(
            agent, state, observations=jnp.ones((2, 1)), rewards=jnp.ones(2),
            next_observations=jnp.ones((2, 1)),
            terminated=jnp.asarray([False, True]), actions=actions,
        )

    expected = run(jnp.asarray([1, 100], dtype=jnp.int32))
    actions = jnp.asarray([1, 100], dtype=dtype)
    for result in (run(actions), jax.jit(run)(actions)):
        np.testing.assert_array_equal(result.updates_applied, [True, True])
        for actual, reference in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
            if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
                actual, reference = jax.random.key_data(actual), jax.random.key_data(reference)
            np.testing.assert_array_equal(actual, reference)
