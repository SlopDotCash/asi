"""Hostile-identity tests for recurring artifact validation consumer."""

from __future__ import annotations

from alberta_framework.evaluation.recurring_feature_artifact import (
    _finite_number,
    _pair_set,
    validate_recurring_feature_artifact,
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

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")


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

    def __len__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __len__")


def test_validate_rejects_hostile_mapping_without_dispatch() -> None:
    hostile = _HostileMapping(
        {
            "schema_version": "test",
            "scientific_payload": {},
            "scientific_digest": {},
            "operational_metadata": {},
        }
    )
    _HostileMapping.calls = 0
    result = validate_recurring_feature_artifact(hostile)  # type: ignore[arg-type]
    assert not result.valid
    assert _HostileMapping.calls == 0


def test_finite_number_rejects_hostile_int_without_float_dispatch() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    assert _finite_number(hostile) is None
    assert _HostileInt.calls == 0


def test_pair_set_rejects_hostile_list_without_iter_dispatch() -> None:
    hostile = _HostileList([[0, 1], [0, 2]])
    _HostileList.calls = 0
    assert _pair_set(hostile) is None
    assert _HostileList.calls == 0
    _HostileList.calls = 0
    assert _pair_set(_HostileList([[0, 1]])) is None
    assert _HostileList.calls == 0
