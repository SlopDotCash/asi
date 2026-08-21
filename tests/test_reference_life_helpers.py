"""Unit coverage for reference_life_scorecard pure helpers.

Fills the two uncovered primitives: exact JSON equality (no bool/int
coercion, byte-exact float comparison) and the scorecard lifecycle-id
format check.
"""


from alberta_framework.benchmarks.reference_life_scorecard import (
    _is_scorecard_lifecycle_id,
    _json_exact_equal,
)


def test_json_exact_equal_dicts() -> None:
    assert _json_exact_equal({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1}) is True
    assert _json_exact_equal({"a": 1}, {"a": 2}) is False
    assert _json_exact_equal({"a": 1}, {"a": 1, "b": 2}) is False


def test_json_exact_equal_lists() -> None:
    assert _json_exact_equal([1, 2, 3], [1, 2, 3]) is True
    assert _json_exact_equal([1, 2], [2, 1]) is False
    assert _json_exact_equal([1, 2], [1, 2, 3]) is False


def test_json_exact_equal_bool_int_distinct() -> None:
    # Python treats True == 1; canonical JSON equality must not.
    assert _json_exact_equal(True, 1) is False
    assert _json_exact_equal(False, 0) is False
    assert _json_exact_equal(True, True) is True
    assert _json_exact_equal(1, 1) is True


def test_json_exact_equal_float_bytes() -> None:
    import struct

    # NaN must compare by bit pattern (both are NaN but bytes equal).
    nan1 = struct.unpack(">d", struct.pack(">d", float("nan")))[0]
    assert _json_exact_equal(nan1, nan1) is True
    # Different values differ.
    assert _json_exact_equal(1.5, 1.25) is False
    # int vs float distinct.
    assert _json_exact_equal(1, 1.0) is False


def test_json_exact_equal_nested_mixed() -> None:
    assert _json_exact_equal({"a": [{"b": True}]}, {"a": [{"b": True}]}) is True
    assert _json_exact_equal({"a": [{"b": True}]}, {"a": [{"b": 1}]}) is False


def test_lifecycle_id_valid() -> None:
    assert _is_scorecard_lifecycle_id("prototype.0123456789abcdef") is True
    assert _is_scorecard_lifecycle_id("prototype." + "a" * 16) is True


def test_lifecycle_id_invalid() -> None:
    assert _is_scorecard_lifecycle_id("prototype.0123456789abcd") is False  # 15 chars
    assert _is_scorecard_lifecycle_id("prototype.0123456789abcdeg") is False  # non-hex
    assert _is_scorecard_lifecycle_id("other.0123456789abcdef") is False  # wrong prefix
    assert _is_scorecard_lifecycle_id(123) is False  # not a string
    assert _is_scorecard_lifecycle_id("") is False
