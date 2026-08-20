"""Canonical JSON configs reject oversized width before the 1 MiB encoding walk."""

from __future__ import annotations

import json

import pytest

from alberta_framework.reference_agent import (
    _MAX_CONFIG_BYTES,
    _MAX_JSON_VALUES,
    _canonical_json_bytes,
    _canonical_json_string_size,
    _validate_json_value,
    canonical_config_sha256,
)


class _HostileList(list[object]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile list hook ran")


class _HostileDict(dict[str, object]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile dict hook ran")


def test_json_value_bound_is_derived_from_canonical_byte_cap() -> None:
    assert _MAX_JSON_VALUES == 524_287


@pytest.mark.parametrize("value", ['"\\\n\u007f😀', "ascii", "é", "\ud800"])
def test_canonical_string_size_matches_ensure_ascii_encoding(value: str) -> None:
    expected = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    assert _canonical_json_string_size(value, path="config") == len(expected)


def test_last_fit_json_list_uses_the_exact_canonical_byte_cap() -> None:
    payload: dict[str, object] = {"": [0] * (_MAX_JSON_VALUES - 2)}
    nodes = _validate_json_value(payload, path="config")
    assert nodes == _MAX_JSON_VALUES
    assert len(_canonical_json_bytes(payload)) == _MAX_CONFIG_BYTES


@pytest.mark.parametrize("count", [_MAX_JSON_VALUES - 1, _MAX_JSON_VALUES, 1_000_000])
def test_canonical_config_rejects_oversized_list_before_children(count: int) -> None:
    payload = {"": [object(), *([0] * (count - 1))]}
    with pytest.raises(ValueError, match="JSON value resource limit"):
        canonical_config_sha256(payload)


def test_canonical_config_rejects_pointer_repeated_nested_lists() -> None:
    row = [0] * 1_000
    payload: dict[str, object] = {"k": [row] * 1_000}
    with pytest.raises(ValueError, match="JSON value resource limit"):
        canonical_config_sha256(payload)


def test_canonical_config_rejects_repeated_large_strings_before_encoding() -> None:
    chunk = "x" * 600_000
    with pytest.raises(ValueError, match="canonical config exceeds"):
        canonical_config_sha256({"k": [chunk, chunk]})


def test_canonical_config_rejects_container_subclasses_before_hooks() -> None:
    hostile_list = _HostileList()
    hostile_dict = _HostileDict()
    _HostileList.calls = 0
    _HostileDict.calls = 0
    with pytest.raises(ValueError, match="not a canonical JSON value"):
        _validate_json_value({"k": hostile_list}, path="config")
    with pytest.raises(ValueError, match="exact JSON object"):
        canonical_config_sha256(hostile_dict)
    assert _HostileList.calls == 0
    assert _HostileDict.calls == 0
