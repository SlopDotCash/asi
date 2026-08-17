"""Shared strict JSON loading for integrity-sensitive object artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NoReturn

_CONCRETE_PATH_TYPE = type(Path())
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_VALUES = 1_000_000


def _require_path(value: object, *, name: str = "path") -> Path:
    if type(value) is str:
        return Path(value)
    if type(value) is _CONCRETE_PATH_TYPE:
        return Path(value)
    raise ValueError(f"{name} must be an exact str or Path")


def _reject_nonstandard_constant(value: str) -> NoReturn:
    if type(value) is not str:
        raise ValueError("non-standard JSON numeric constant must be an exact string")
    raise ValueError("non-standard JSON numeric constant is forbidden")


def _parse_finite_float(value: str) -> float:
    if type(value) is not str:
        raise ValueError("JSON float token must be an exact string")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is forbidden")
    return parsed


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    if type(pairs) is not list:
        raise ValueError("JSON object pairs must be an exact list")
    parsed: dict[str, Any] = {}
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("JSON object entries must be exact key-value pairs")
        key, value = pair
        if type(key) is not str:
            raise ValueError("JSON object keys must be exact strings")
        if key in parsed:
            raise ValueError("duplicate JSON object key")
        parsed[key] = value
    return parsed


def _decode_bounded_utf8_json(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
    except OSError:
        raise
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("JSON payload exceeds the byte limit")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("JSON payload must be valid UTF-8") from error

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                raise ValueError("JSON payload exceeds the nesting-depth limit")
        elif character in "]}":
            depth -= 1
    return text


def _validate_exact_json_tree(root: object) -> None:
    stack: list[tuple[object, int]] = [(root, 1)]
    values_seen = 0
    while stack:
        value, depth = stack.pop()
        values_seen += 1
        if values_seen > _MAX_JSON_VALUES:
            raise ValueError("JSON payload exceeds the value-count limit")
        if depth > _MAX_JSON_DEPTH:
            raise ValueError("JSON payload exceeds the nesting-depth limit")
        if type(value) is dict:
            for key, child in value.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be exact strings")
                stack.append((child, depth + 1))
        elif type(value) is list:
            stack.extend((child, depth + 1) for child in value)
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("non-finite JSON number is forbidden")
        elif type(value) not in (str, int, bool, type(None)):
            raise ValueError("JSON payload contains a non-JSON value")


def load_strict_json_object(path: Path | str) -> dict[str, Any]:
    """Load one finite UTF-8 JSON object, rejecting duplicate keys at every depth."""

    resolved = _require_path(path, name="path")
    text = _decode_bounded_utf8_json(resolved)
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_nonstandard_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except RecursionError as error:
        raise ValueError("JSON payload exceeds the parser recursion limit") from error
    _validate_exact_json_tree(parsed)
    if type(parsed) is not dict:
        raise ValueError(f"{resolved}: payload must contain one JSON object")
    return parsed
