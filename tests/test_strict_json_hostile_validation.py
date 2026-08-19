"""Hostile-safe validation for strict JSON loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import alberta_framework._strict_json as strict_json
from alberta_framework._strict_json import (
    _reject_duplicate_object_keys,
    _validate_exact_json_tree,
    load_strict_json_object,
    load_strict_json_object_from_text,
)


class _StringSubclass(str):
    pass


class _HostilePath(Path):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("str hook")

    def __fspath__(self) -> str:  # pragma: no cover
        raise AssertionError("fspath hook")


def test_load_rejects_string_subclass_path(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="exact str or Path"):
        load_strict_json_object(_StringSubclass(str(good)))


def test_load_rejects_hostile_path() -> None:
    hostile = object.__new__(_HostilePath)
    with pytest.raises(ValueError, match="exact str or Path"):
        load_strict_json_object(hostile)


def test_load_rejects_hostile_path_via_str_subclass() -> None:
    with pytest.raises(ValueError, match="exact str or Path"):
        load_strict_json_object(_StringSubclass("/tmp/x.json"))


def test_load_with_exact_str_path(tmp_path: Path) -> None:
    payload = {"a": 1, "b": 2}
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_strict_json_object(str(path)) == payload


def test_load_with_posix_path(tmp_path: Path) -> None:
    payload = {"a": 1}
    path = tmp_path / "ok2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_strict_json_object(path) == payload


def test_load_rejects_non_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "arr.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="payload must contain one JSON object"):
        load_strict_json_object(path)


def test_reject_duplicate_does_not_invoke_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

        def __hash__(self) -> int:  # pragma: no cover
            raise RuntimeError("hash hook")

    evil = EvilStr("dup")
    with pytest.raises(ValueError, match="exact strings"):
        _reject_duplicate_object_keys([(evil, 1), (evil, 2)])


def test_reject_duplicate_rejects_string_subclass_key() -> None:
    key = _StringSubclass("dup")
    with pytest.raises(ValueError, match="exact strings"):
        _reject_duplicate_object_keys([(key, 1), (key, 2)])


def test_load_rejects_duplicate_keys_in_file(tmp_path: Path) -> None:
    path = tmp_path / "dup.json"
    # duplicate key "a" at object level
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_strict_json_object(path)


def test_load_rejects_nonfinite_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
        load_strict_json_object(path)


def test_load_from_text_rejects_duplicate_keys_and_non_strings() -> None:
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_strict_json_object_from_text('{"a": 1, "a": 2}', label="metadata")
    with pytest.raises(ValueError, match="must be an exact string"):
        load_strict_json_object_from_text(_StringSubclass("{}"), label="metadata")
    with pytest.raises(ValueError, match="must be an exact string"):
        load_strict_json_object_from_text(b"{}", label="metadata")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty exact string"):
        load_strict_json_object_from_text("{}", label="")


def test_load_from_text_rejects_overflow_exponent() -> None:
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_strict_json_object_from_text('{"value":1e999}', label="metadata")


def test_hostile_path_not_invoke_fspath_on_error() -> None:
    hostile = object.__new__(_HostilePath)
    try:
        load_strict_json_object(hostile)
    except ValueError as exc:
        assert "exact str or Path" in str(exc)
        # ensure hook not invoked — would have raised AssertionError
    else:
        raise AssertionError("should have raised")


def test_load_rejects_bytes_path() -> None:
    with pytest.raises(ValueError, match="exact str or Path"):
        load_strict_json_object(b"/tmp/x.json")  # type: ignore[arg-type]


def test_load_rejects_excessive_depth_before_parser_recursion(tmp_path: Path) -> None:
    path = tmp_path / "deep.json"
    path.write_text('{"x":' * 65 + "0" + "}" * 65, encoding="utf-8")
    with pytest.raises(ValueError, match="nesting-depth"):
        load_strict_json_object(path)


def test_load_rejects_oversized_payload_with_bounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(strict_json, "_MAX_JSON_BYTES", 16)
    path = tmp_path / "large.json"
    path.write_bytes(b'{"value":"' + b"x" * 32 + b'"}')
    with pytest.raises(ValueError, match="byte limit"):
        load_strict_json_object(path)


def test_exact_tree_validator_rejects_non_json_containers_and_values() -> None:
    with pytest.raises(ValueError, match="non-JSON value"):
        _validate_exact_json_tree({"value": (1, 2)})
    with pytest.raises(ValueError, match="exact strings"):
        _validate_exact_json_tree({_StringSubclass("key"): 1})
    with pytest.raises(ValueError, match="non-finite"):
        _validate_exact_json_tree({"value": float("inf")})


def test_duplicate_hook_rejects_non_string_key_before_hostile_hooks() -> None:
    class HostileKey:
        def __hash__(self) -> int:
            raise AssertionError("hash hook executed")

        def __repr__(self) -> str:
            raise AssertionError("repr hook executed")

    with pytest.raises(ValueError, match="exact strings"):
        _reject_duplicate_object_keys([(HostileKey(), 1)])  # type: ignore[list-item]
