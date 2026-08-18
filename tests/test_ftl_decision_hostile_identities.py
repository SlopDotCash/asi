"""Hostile-identity tests for ftl decision artifact consumer."""

from __future__ import annotations

from alberta_framework.evaluation.ftl_decision_artifact import (
    _finite_number,
    _mapping,
    validate_ftl_decision_artifact,
)


class _HostileMapping(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __getitem__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")

    def __contains__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __contains__")


class _HostileInt(int):
    calls = 0

    def __float__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __float__")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")


class _HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")


def test_mapping_rejects_hostile_without_dispatch() -> None:
    hostile_value = _HostileMapping({"a": 1})
    parent: dict[str, object] = {"key": hostile_value}
    _HostileMapping.calls = 0
    result = _mapping(parent, "key", "test", [])
    assert result is None
    assert _HostileMapping.calls == 0


def test_finite_number_rejects_hostile_int() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    assert _finite_number(hostile) is None
    assert _HostileInt.calls == 0


def test_validate_rejects_hostile_mapping_without_iter() -> None:
    hostile = _HostileMapping(
        {
            "schema_version": "alberta.ftl_decision.v1",
            "scientific_payload": {},
            "scientific_digest": {},
            "operational_metadata": {},
        }
    )
    _HostileMapping.calls = 0
    result = validate_ftl_decision_artifact(hostile)  # type: ignore[arg-type]
    assert not result.valid
    assert _HostileMapping.calls == 0


def test_compare_structure_rejects_hostile_list() -> None:
    from alberta_framework.evaluation.ftl_decision_artifact import _compare_structure

    hostile = _HostileList([1, 2])
    _HostileList.calls = 0
    errors: list[str] = []
    # expected is builtin list, actual is hostile list
    _compare_structure(hostile, [1, 2], "test", errors)
    assert errors
    assert _HostileList.calls == 0
