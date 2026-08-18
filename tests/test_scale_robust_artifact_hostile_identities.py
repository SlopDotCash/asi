"""Hostile-identity tests for scale-robust artifact validation consumer."""

from __future__ import annotations

from alberta_framework.evaluation.scale_robust_feature_artifact import (
    _finite_number,
    _parsed_pairs,
    _strict_int,
    validate_evidence_artifact,
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

    def keys(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile keys")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")

    def __eq__(self, other):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __eq__")


class _HostileInt(int):
    calls = 0

    def __float__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __float__")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")

    def __repr__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __repr__")


class _HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __len__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __len__")

    def __getitem__(self, index):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")

    def __hash__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __hash__")


def test_validate_rejects_hostile_mapping_without_dispatch() -> None:
    hostile = _HostileMapping(
        {
            "schema_version": "test",
            "scientific_payload": {},
            "content_digest": {},
            "operational_metadata": {},
        }
    )
    _HostileMapping.calls = 0
    result = validate_evidence_artifact(hostile)  # type: ignore[arg-type]
    assert not result.valid
    assert _HostileMapping.calls == 0


def test_finite_number_rejects_hostile_int_without_float_dispatch() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    assert _finite_number(hostile) is None
    assert _HostileInt.calls == 0
    # Use a minimal valid-like structure: seed_records is validated, but we can
    # test that hostile int inside phase_windows is rejected without dispatch.
    # Directly validate via _finite_number is the precise hostile-float boundary.
    _HostileInt.calls = 0
    assert _strict_int(hostile) is None
    assert _HostileInt.calls == 0


def test_parsed_pairs_rejects_hostile_list_without_iter_dispatch() -> None:
    hostile = _HostileList([(0, 1), (0, 2)])
    _HostileList.calls = 0
    assert _parsed_pairs(hostile) is None
    assert _HostileList.calls == 0
    # Hostile inner is not iterated because outer is hostile and rejected at top-level.
    _HostileList.calls = 0
    assert _parsed_pairs(_HostileList([[0, 1]])) is None
    assert _HostileList.calls == 0
