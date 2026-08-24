"""Tests for fail-closed host identities on discrete-action safety."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.update_safety import masked_convex_weights, safe_discrete_action


def test_legal_action_in_domain_is_valid() -> None:
    safe, valid = safe_discrete_action(1, 3, allow_unset=False)
    assert int(safe) == 1
    assert bool(valid)


def test_legal_unset_sentinel_is_valid() -> None:
    safe, valid = safe_discrete_action(-1, 3, allow_unset=True)
    assert int(safe) == -1
    assert bool(valid)


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


_SMALLEST_NORMAL_FLOAT32 = float(np.finfo(np.float32).tiny)
# A softmax over these logit gaps puts the two trailing entries far enough below
# the leader to walk the masked mass from ordinary down through the old 1e-12
# floor to an exact zero.
_MASS_DECAY_GAPS = [0.0, 10.0, 28.0, 30.0, 40.0, 60.0, 80.0, 87.0, 88.0, 110.0, 200.0]


def _trailing_pair_allocation(gap: float) -> tuple[Array, Array]:
    """Return ``(mask, weights)`` for a softmax whose leader is ``gap`` nats ahead."""

    weights = jax.nn.softmax(jnp.asarray([gap, 0.0, 0.0], dtype=jnp.float32))
    return jnp.asarray([False, True, True]), weights


@pytest.mark.parametrize("gap", _MASS_DECAY_GAPS)
def test_masked_convex_weights_sum_to_one_over_the_mask(gap: float) -> None:
    mask, weights = _trailing_pair_allocation(gap)
    row = masked_convex_weights(mask, weights)
    np.testing.assert_allclose(float(jnp.sum(row)), 1.0, rtol=0.0, atol=6e-8)


@pytest.mark.parametrize("gap", _MASS_DECAY_GAPS)
def test_masked_convex_weights_zero_the_masked_out_entries(gap: float) -> None:
    mask, weights = _trailing_pair_allocation(gap)
    row = np.asarray(masked_convex_weights(mask, weights))
    assert row[0] == 0.0


@pytest.mark.parametrize("gap", _MASS_DECAY_GAPS)
def test_masked_convex_weights_bound_the_baseline_by_the_values_averaged(gap: float) -> None:
    mask, weights = _trailing_pair_allocation(gap)
    row = masked_convex_weights(mask, weights)
    values = jnp.asarray([0.0, 1.0, 3.0], dtype=jnp.float32)
    baseline = float(jnp.sum(row * values))
    assert 1.0 - 6e-8 <= baseline <= 3.0 + 6e-8


@pytest.mark.parametrize("shift", [0.0, 1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("gap", _MASS_DECAY_GAPS)
def test_masked_convex_weights_make_a_uniform_shift_move_the_baseline_by_that_shift(
    gap: float, shift: float
) -> None:
    mask, weights = _trailing_pair_allocation(gap)
    row = masked_convex_weights(mask, weights)
    values = jnp.asarray([0.0, 1.0, 3.0], dtype=jnp.float32)
    baseline = float(jnp.sum(row * values))
    shifted = float(jnp.sum(row * (values + shift)))
    np.testing.assert_allclose(shifted - baseline, shift, rtol=0.0, atol=1e-4)


def test_masked_convex_weights_match_the_plain_quotient_for_a_normal_mass() -> None:
    mask, weights = _trailing_pair_allocation(40.0)
    row = np.asarray(masked_convex_weights(mask, weights))
    reference = np.asarray(jnp.where(mask, weights, 0.0)) / float(
        jnp.sum(jnp.where(mask, weights, 0.0))
    )
    np.testing.assert_array_equal(row, reference)


def test_masked_convex_weights_fall_back_to_uniform_when_the_mass_underflows() -> None:
    mask = jnp.asarray([True, False, True, True])
    weights = jnp.zeros(4, dtype=jnp.float32)
    row = np.asarray(masked_convex_weights(mask, weights))
    third = 1.0 / 3.0
    np.testing.assert_array_equal(row, np.asarray([third, 0.0, third, third], dtype=np.float32))


def test_masked_convex_weights_fall_back_to_uniform_for_a_subnormal_mass() -> None:
    mask = jnp.asarray([True, True, False])
    weights = jnp.asarray([_SMALLEST_NORMAL_FLOAT32 / 4.0, 0.0, 1.0], dtype=jnp.float32)
    row = np.asarray(masked_convex_weights(mask, weights))
    np.testing.assert_array_equal(row, np.asarray([0.5, 0.5, 0.0], dtype=np.float32))


def test_masked_convex_weights_return_zeros_for_an_empty_mask() -> None:
    mask = jnp.zeros(3, dtype=jnp.bool_)
    weights = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)
    row = np.asarray(masked_convex_weights(mask, weights))
    np.testing.assert_array_equal(row, np.zeros(3, dtype=np.float32))


def test_masked_convex_weights_pass_a_non_finite_allocation_through() -> None:
    mask = jnp.asarray([True, True])
    weights = jnp.asarray([jnp.nan, jnp.nan], dtype=jnp.float32)
    row = np.asarray(masked_convex_weights(mask, weights))
    assert not np.isfinite(row).any()


@pytest.mark.parametrize("gap", _MASS_DECAY_GAPS)
def test_masked_convex_weights_stay_finite_and_non_negative(gap: float) -> None:
    mask, weights = _trailing_pair_allocation(gap)
    row = np.asarray(masked_convex_weights(mask, weights))
    assert np.isfinite(row).all()
    assert (row >= 0.0).all()


@pytest.mark.parametrize("gap", _MASS_DECAY_GAPS)
def test_masked_convex_weights_agree_under_jit(gap: float) -> None:
    mask, weights = _trailing_pair_allocation(gap)
    eager = np.asarray(masked_convex_weights(mask, weights))
    compiled = np.asarray(jax.jit(masked_convex_weights)(mask, weights))
    np.testing.assert_array_equal(eager, compiled)

