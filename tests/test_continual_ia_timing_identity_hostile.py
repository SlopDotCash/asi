"""Hostile identity gates for operational condition-timing validation."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia_artifact import _validate_operational

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile int equality hook executed")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile int hash hook executed")


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile str equality hook executed")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile str hash hook executed")


def _operational(timings: list[dict[str, object]]) -> dict[str, object]:
    return {
        "digest_exclusion_reason": (
            "host environment and wall-clock timing are non-deterministic"
        ),
        "environment": None,
        "condition_timings": timings,
        "overall_acceptance_passed": True,
    }


def _timing(seed: object, condition: object) -> dict[str, object]:
    return {
        "seed": seed,
        "condition": condition,
        "wall_seconds": 1.0,
        "mean_step_latency_ms": 1.0,
    }


def _identity_errors(errors: list[str]) -> list[str]:
    return [error for error in errors if "has invalid identity" in error]


class TestConditionTimingIdentity:
    def test_exact_int_seed_and_exact_str_condition_pass(self) -> None:
        errors: list[str] = []
        _validate_operational(
            _operational([_timing(30, "current")]),
            expected_results=None,
            expected_acceptance=True,
            errors=errors,
        )
        assert _identity_errors(errors) == []

    def test_bool_seed_is_rejected(self) -> None:
        errors: list[str] = []
        _validate_operational(
            _operational([_timing(True, "current")]),
            expected_results=None,
            expected_acceptance=True,
            errors=errors,
        )
        assert _identity_errors(errors) == [
            "condition_timings[0] has invalid identity"
        ]

    def test_int_subclass_seed_is_rejected_without_hooks(self) -> None:
        _HostileInt.calls = 0
        errors: list[str] = []
        _validate_operational(
            _operational([_timing(_HostileInt(30), "current")]),
            expected_results=None,
            expected_acceptance=True,
            errors=errors,
        )
        assert _identity_errors(errors) == [
            "condition_timings[0] has invalid identity"
        ]
        assert _HostileInt.calls == 0

    def test_str_subclass_condition_is_rejected_without_hooks(self) -> None:
        _HostileStr.calls = 0
        errors: list[str] = []
        _validate_operational(
            _operational([_timing(30, _HostileStr("current"))]),
            expected_results=None,
            expected_acceptance=True,
            errors=errors,
        )
        assert _identity_errors(errors) == [
            "condition_timings[0] has invalid identity"
        ]
        assert _HostileStr.calls == 0
