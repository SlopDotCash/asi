"""Hostile validation for matched Alberta worker facade."""

import pytest

from alberta_framework.benchmarks._forager_matched_alberta_worker import (
    MatchedAlbertaWorkerError,
    _parse_agent_configuration,
    _parse_finite_json_float,
    _reject_duplicate_keys,
    _reject_nonfinite,
)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")

    def lower(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.lower must not be called")


class _StringSubclass(str):
    pass


def test_reject_duplicate_keys_hostile_without_repr_leak() -> None:
    evil = _EvilStr("dup")
    _EvilStr.calls = 0
    with pytest.raises(MatchedAlbertaWorkerError, match="must be a string") as exc:
        _reject_duplicate_keys([("a", 1), (evil, 2)])  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_reject_duplicate_keys_string_subclass_rejected() -> None:
    with pytest.raises(MatchedAlbertaWorkerError, match="must be a string"):
        _reject_duplicate_keys(
            [("a", 1), (_StringSubclass("a"), 2)]  # type: ignore[arg-type]
        )


def test_reject_duplicate_keys_duplicate_without_leak() -> None:
    with pytest.raises(MatchedAlbertaWorkerError, match="duplicate configuration key") as exc:
        _reject_duplicate_keys([("SECRET123", 1), ("SECRET123", 2)])
    assert "!r" not in str(exc.value)
    assert "SECRET123" not in str(exc.value)


def test_reject_nonfinite_hostile_string_without_repr() -> None:
    evil = _EvilStr("NaN")
    _EvilStr.calls = 0
    with pytest.raises(MatchedAlbertaWorkerError, match="must be a string") as exc:
        _reject_nonfinite(evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "EvilStr" not in str(exc.value)


def test_reject_nonfinite_string_subclass_rejected() -> None:
    with pytest.raises(MatchedAlbertaWorkerError, match="must be a string"):
        _reject_nonfinite(_StringSubclass("Infinity"))  # type: ignore[arg-type]


def test_reject_nonfinite_sanitized_message() -> None:
    with pytest.raises(
        MatchedAlbertaWorkerError, match="non-finite configuration number is forbidden"
    ) as exc:
        _reject_nonfinite("NaN")
    assert "!r" not in str(exc.value)
    assert "NaN" not in str(exc.value)


def test_parse_finite_json_float_hostile_without_hook() -> None:
    evil = _EvilStr("1.0")
    _EvilStr.calls = 0
    with pytest.raises(MatchedAlbertaWorkerError, match="must be a string") as exc:
        _parse_finite_json_float(evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "EvilStr" not in str(exc.value)


def test_parse_finite_json_float_string_subclass_rejected() -> None:
    with pytest.raises(MatchedAlbertaWorkerError, match="must be a string"):
        _parse_finite_json_float(_StringSubclass("1.0"))  # type: ignore[arg-type]


def test_parse_finite_json_float_nonfinite_sanitized() -> None:
    with pytest.raises(
        MatchedAlbertaWorkerError, match="non-finite configuration number is forbidden"
    ) as exc:
        _parse_finite_json_float("inf")
    assert "!r" not in str(exc.value)


def test_parse_finite_json_float_parses_finite() -> None:
    assert _parse_finite_json_float("1.5") == 1.5


def test_parse_agent_configuration_hostile_implementation_kind() -> None:
    evil = _EvilStr("alberta_causal_map")
    _EvilStr.calls = 0
    with pytest.raises(MatchedAlbertaWorkerError, match="must be a string") as exc:
        _parse_agent_configuration(evil, {})  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "EvilStr" not in str(exc.value)


def test_parse_agent_configuration_string_subclass_rejected() -> None:
    with pytest.raises(MatchedAlbertaWorkerError, match="must be a string"):
        _parse_agent_configuration(
            _StringSubclass("alberta_causal_map"), {}  # type: ignore[arg-type]
        )


def test_parse_agent_configuration_unsupported_sanitized() -> None:
    with pytest.raises(
        MatchedAlbertaWorkerError, match="unsupported Alberta implementation kind"
    ) as exc:
        _parse_agent_configuration("unknown_kind", {})
    assert "!r" not in str(exc.value)
    assert "unknown_kind" not in str(exc.value)
