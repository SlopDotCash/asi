"""Shared strict JSON loading for integrity-sensitive object artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NoReturn


def _require_path(value: object, *, name: str = "path") -> Path:
    if type(value) is str:
        return Path(value)
    if isinstance(value, Path) and type(value).__module__.startswith("pathlib"):
        return Path(value)
    raise ValueError(f"{name} must be an exact str or Path")


def _reject_nonstandard_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key: {key}")
        parsed[key] = value
    return parsed


def load_strict_json_object(path: Path | str) -> dict[str, Any]:
    """Load one finite UTF-8 JSON object, rejecting duplicate keys at every depth."""

    resolved = _require_path(path, name="path")
    parsed = json.loads(
        resolved.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
        parse_float=_parse_finite_float,
        object_pairs_hook=_reject_duplicate_object_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"{resolved}: payload must contain one JSON object")
    return parsed
