"""Unit coverage for alberta_framework.steps.step5.

Tests the fail-closed validation primitives: integer gates (type/bounds),
built-in bool, float32 storage compatibility, and the acyclic closed-JSON
object check (cycles, keys, finite floats, exact identities).
"""

import pytest

from alberta_framework.steps.step5 import (
    _compatible_float32_storage,
    _require_bool,
    _require_closed_json_object,
    _require_int,
)


def test_require_int_accepts() -> None:
    assert _require_int("x", 5) == 5
    assert _require_int("x", 0, minimum=0) == 0
    assert _require_int("x", 10, maximum=10) == 10


def test_require_int_rejects_bool() -> None:
    with pytest.raises(ValueError, match="integer"):
        _require_int("x", True)


def test_require_int_rejects_float() -> None:
    with pytest.raises(ValueError, match="integer"):
        _require_int("x", 1.5)


def test_require_int_bounds() -> None:
    with pytest.raises(ValueError, match="positive"):
        _require_int("x", 0, minimum=1)
    with pytest.raises(ValueError, match="non-negative"):
        _require_int("x", -1, minimum=0)
    with pytest.raises(ValueError, match=">="):
        _require_int("x", 2, minimum=3)
    with pytest.raises(ValueError, match="<="):
        _require_int("x", 11, maximum=10)


def test_require_bool() -> None:
    assert _require_bool("x", True) is True
    with pytest.raises(ValueError, match="built-in bool"):
        _require_bool("x", 1)
    with pytest.raises(ValueError, match="built-in bool"):
        _require_bool("x", "true")


def test_compatible_float32_storage() -> None:
    assert _compatible_float32_storage(1.5, 1.5) == 1.5  # float kept as-is
    assert _compatible_float32_storage(3, 3.0) == 3  # int equal to narrowed kept
    assert _compatible_float32_storage(3, 3.5) == 3.5  # mismatch → narrowed


def test_closed_json_object_valid() -> None:
    assert _require_closed_json_object("x", {"a": [1, 2.5, "s", True, None]}) == {
        "a": [1, 2.5, "s", True, None]
    }


def test_closed_json_object_rejects_cycle() -> None:
    a = {}
    a["self"] = a
    with pytest.raises(ValueError, match="acyclic"):
        _require_closed_json_object("x", a)


def test_closed_json_object_rejects_bad_keys() -> None:
    with pytest.raises(ValueError, match="exact strings"):
        _require_closed_json_object("x", {1: "a"})


def test_closed_json_object_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite"):
        _require_closed_json_object("x", {"a": float("nan")})


def test_closed_json_object_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="exact JSON identities"):
        _require_closed_json_object("x", {"a": object()})
