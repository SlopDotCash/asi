"""Hostile-identity tests for forager rng parity consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_rng_parity import (
    ForagerRngParityError,
    _require_exact_keys,
    _require_object,
    validate_parity_result,
)


class _HostileMapping(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __getitem__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")

    def values(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile values")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")

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


def test_require_object_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ForagerRngParityError):
        _require_object(hostile, "test")
    assert _HostileMapping.calls == 0


def test_require_exact_keys_rejects_hostile_without_iter() -> None:
    hostile = _HostileMapping({"a": 1, "b": 2})
    _HostileMapping.calls = 0
    with pytest.raises(ForagerRngParityError):
        _require_exact_keys(hostile, {"a", "b"}, "test")
    assert _HostileMapping.calls == 0


def test_validate_parity_result_rejects_hostile_top_level() -> None:
    hostile = _HostileMapping(
        {
            "schema_version": "alberta.forager_rng_parity.v1",
            "status": "matched",
            "evidence_boundary": "test",
            "promotion_authorized": False,
            "runtime": {},
            "task": {},
            "rng_contract": {},
            "probe": {},
            "matched_trace": {},
            "wrapper_trace_sha256": "0" * 64,
            "direct_trace_sha256": "0" * 64,
            "payload_sha256": "0" * 64,
        }
    )
    _HostileMapping.calls = 0
    with pytest.raises((ForagerRngParityError, TypeError, ValueError)):
        validate_parity_result(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
