"""Hostile string gate for scorecard resource arm before in."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import (
    _validate_resource_payload,
)

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


def test_resource_arm_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("prototype")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="resource arm is invalid"):
        _validate_resource_payload({}, arm=hostile, path="p")  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_resource_arm_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="resource arm is invalid"):
        _validate_resource_payload({}, arm=123, path="p")  # type: ignore[arg-type]


def test_resource_arm_benign_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        _validate_resource_payload("not_mapping", arm="prototype", path="p")  # type: ignore[arg-type]


def test_resource_arm_benign_valid() -> None:
    # prototype arm maps to prototype method; missing keys not arm error
    try:
        _validate_resource_payload({}, arm="prototype", path="p")
    except ValueError as exc:
        assert "resource arm is invalid" not in str(exc)
    except Exception:
        pass
    # unknown arm falls through to floating_... and should not raise arm invalid
    try:
        _validate_resource_payload({}, arm="unknown_arm", path="p")
    except ValueError as exc:
        assert "resource arm is invalid" not in str(exc)
    except Exception:
        pass
