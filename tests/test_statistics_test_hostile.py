"""Hostile string gate for statistics test before in."""

from __future__ import annotations

import pytest

from alberta_framework.utils.statistics import pairwise_comparisons

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_test_param_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("ttest")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="test is invalid"):
        pairwise_comparisons({}, test=hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_test_param_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="test is invalid"):
        pairwise_comparisons({}, test=123)  # type: ignore[arg-type]


def test_test_param_benign_unknown() -> None:
    # n<2 short-circuits before the test *value* check for exact-str inputs,
    # so benign "unknown" with empty returns {}; hostile non-str is rejected
    # before the short-circuit by the type gate at the top of the function.
    result = pairwise_comparisons({}, test="unknown")
    assert result == {}


def test_test_param_benign_valid_short_circuits() -> None:
    # n<2 returns {} without needing test *value* valid (e.g. "unknown"),
    # but still requires exact str type — hostile/non-str must raise.
    result = pairwise_comparisons({}, test="ttest")
    assert result == {}
    result = pairwise_comparisons({}, test="mann_whitney")
    assert result == {}
