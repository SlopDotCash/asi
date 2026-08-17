"""Tests for fail-closed host identities on discrete-action safety."""

from __future__ import annotations

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
