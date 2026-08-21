"""Unit coverage for alberta_framework._scan_resources.

Tests the host-side scan-budget contracts: budget validation, step and
parallel counts, step-unit products, and JAX leading-length checks.
"""

import jax
import jax.numpy as jnp
import pytest

from alberta_framework._scan_resources import (
    ScanBudget,
    require_jax_leading_length,
    require_matching_jax_leading_length,
    require_parallel_count,
    require_scan_steps,
    require_step_units,
)


def _budget(**kwargs):
    defaults = {"label": "test", "maximum_steps": 100, "maximum_parallel": 4}
    defaults.update(kwargs)
    return ScanBudget(**defaults)


def test_budget_valid() -> None:
    b = _budget()
    assert b.maximum_steps == 100


def test_budget_rejects_bad_fields() -> None:
    with pytest.raises(ValueError, match="label"):
        _budget(label="")
    with pytest.raises(ValueError, match="maximum_steps"):
        _budget(maximum_steps=0)
    with pytest.raises(ValueError, match="maximum_steps"):
        _budget(maximum_steps=1.5)
    with pytest.raises(ValueError, match="maximum_parallel"):
        _budget(maximum_parallel=0)
    with pytest.raises(ValueError, match="maximum_step_units"):
        _budget(maximum_step_units=0)


def test_require_scan_steps() -> None:
    b = _budget()
    assert require_scan_steps("n", 50, b) == 50
    with pytest.raises(ValueError, match="\\[1, 100\\]"):
        require_scan_steps("n", 0, b)
    with pytest.raises(ValueError, match="\\[1, 100\\]"):
        require_scan_steps("n", 101, b)
    with pytest.raises(ValueError, match="\\[1, 100\\]"):
        require_scan_steps("n", 2.5, b)


def test_require_parallel_count() -> None:
    b = _budget()
    assert require_parallel_count("p", 4, b) == 4
    with pytest.raises(ValueError, match="\\[1, 4\\]"):
        require_parallel_count("p", 5, b)


def test_require_step_units() -> None:
    b = _budget(maximum_step_units=200)
    require_step_units(50, 4, b)  # 200 == max ok
    with pytest.raises(ValueError, match="step-units"):
        require_step_units(51, 4, b)


def test_require_step_units_no_budget() -> None:
    b = _budget(maximum_step_units=None)
    require_step_units(10**9, 10**9, b)  # no cap → no rejection


def test_require_jax_leading_length() -> None:
    b = _budget(maximum_steps=10)
    arr = jnp.zeros((5, 3))
    assert require_jax_leading_length("x", arr, b) == 5
    with pytest.raises(TypeError, match="JAX array"):
        require_jax_leading_length("x", [[1, 2]], b)
    with pytest.raises(ValueError, match="\\[1, 10\\]"):
        require_jax_leading_length("x", jnp.zeros((11, 3)), b)


def test_require_jax_leading_length_ranks() -> None:
    b = _budget(maximum_steps=10)
    with pytest.raises(ValueError, match="rank in"):
        require_jax_leading_length("x", jnp.zeros((5, 3)), b, ranks=(1,))


def test_require_matching_leading_length() -> None:
    arr = jnp.zeros((5, 3))
    require_matching_jax_leading_length("y", arr, expected=5)
    with pytest.raises(ValueError, match="share the primary"):
        require_matching_jax_leading_length("y", arr, expected=4)
