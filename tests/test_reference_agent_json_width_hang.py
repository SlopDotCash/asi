"""Canonical JSON configs enforce resource bounds before encoding."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

import alberta_framework.reference_agent as reference_agent
from alberta_framework.reference_agent import (
    _MAX_CONFIG_BYTES,
    _MAX_JSON_VALUES,
    MAX_DECISION_INDEX,
    _validate_json_value,
    canonical_config_sha256,
)


def test_frozen_json_value_bound_matches_strict_json_last_fit() -> None:
    assert _MAX_JSON_VALUES == 1_000_000


def test_json_value_counter_accepts_small_builtin_containers() -> None:
    assert _validate_json_value({"k": [0, None, True]}, path="config") == 5


def test_canonical_config_rejects_oversized_width_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encoding_must_not_run(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized config reached json.dumps")

    monkeypatch.setattr(json, "dumps", encoding_must_not_run)
    count = _MAX_CONFIG_BYTES // 2 + 1
    with pytest.raises(ValueError, match="canonical config exceeds"):
        canonical_config_sha256({"k": [0] * count})


@pytest.mark.parametrize(
    "payload",
    [
        {"k": "x" * (_MAX_CONFIG_BYTES + 1)},
        {"x" * (_MAX_CONFIG_BYTES + 1): 0},
    ],
)
def test_canonical_config_rejects_oversized_text_before_encoding(
    payload: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def encoding_must_not_run(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized config reached json.dumps")

    monkeypatch.setattr(json, "dumps", encoding_must_not_run)
    with pytest.raises(ValueError, match="canonical config exceeds"):
        canonical_config_sha256(payload)


def test_canonical_config_counts_escaped_string_bytes_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encoding_must_not_run(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized config reached json.dumps")

    monkeypatch.setattr(json, "dumps", encoding_must_not_run)
    payload = {"k": "\N{GRINNING FACE}" * (_MAX_CONFIG_BYTES // 12 + 1)}
    with pytest.raises(ValueError, match="canonical config exceeds"):
        canonical_config_sha256(payload)


def test_canonical_config_counts_del_escape_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encoding_must_not_run(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized config reached json.dumps")

    monkeypatch.setattr(json, "dumps", encoding_must_not_run)
    count = ((_MAX_CONFIG_BYTES - 8) // 6) + 1
    with pytest.raises(ValueError, match="canonical config exceeds"):
        canonical_config_sha256({"k": "\x7f" * count})


@pytest.mark.parametrize("integer", [1 << 63, MAX_DECISION_INDEX, MAX_DECISION_INDEX + 1])
def test_canonical_config_preserves_large_protocol_integer_domain(integer: int) -> None:
    assert len(canonical_config_sha256({"k": integer})) == 64


def test_canonical_config_rejects_huge_integer_before_decimal_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decimal_conversion_must_not_run(_value: object) -> str:
        raise AssertionError("oversized integer reached decimal conversion")

    monkeypatch.setattr(reference_agent, "str", decimal_conversion_must_not_run, raising=False)
    integer = 1 << (((_MAX_CONFIG_BYTES * 10) // 3) + 2)
    with pytest.raises(ValueError, match="canonical config exceeds"):
        _validate_json_value(integer, path="config.k")


def test_canonical_config_preserves_exact_byte_limit_boundary() -> None:
    assert len(canonical_config_sha256({"k": "x" * (_MAX_CONFIG_BYTES - 8)})) == 64
    with pytest.raises(ValueError, match="canonical config exceeds"):
        canonical_config_sha256({"k": "x" * (_MAX_CONFIG_BYTES - 7)})


class _HostileList(list[object]):
    def __len__(self) -> int:
        raise AssertionError("hostile list length hook ran")

    def __iter__(self) -> Iterator[object]:
        raise AssertionError("hostile list iterator hook ran")


class _HostileMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError("hostile mapping item hook ran")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("hostile mapping iterator hook ran")

    def __len__(self) -> int:
        raise AssertionError("hostile mapping length hook ran")


class _HostileDict(dict[str, object]):
    def __len__(self) -> int:
        raise AssertionError("hostile dict length hook ran")

    def items(self) -> Any:
        raise AssertionError("hostile dict items hook ran")


@pytest.mark.parametrize(
    "payload",
    [{"k": _HostileList()}, _HostileMapping(), _HostileDict()],
)
def test_canonical_config_rejects_hook_bearing_containers_without_dispatch(
    payload: Any,
) -> None:
    with pytest.raises(ValueError, match="canonical JSON value|exact JSON object"):
        canonical_config_sha256(payload)
