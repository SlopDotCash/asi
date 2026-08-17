"""Hostile validation for historical provenance facade."""

import pytest

from alberta_framework.benchmarks.historical_forager_provenance import (
    HISTORICAL_FORAGER_FAMILY_ID,
    HistoricalForagerFamilyMismatchError,
    HistoricalForagerProvenanceError,
    assert_historical_family_pairing,
)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


def test_hostile_left_without_repr_leak() -> None:
    evil = _EvilStr(HISTORICAL_FORAGER_FAMILY_ID)
    _EvilStr.calls = 0
    with pytest.raises(HistoricalForagerProvenanceError, match="must be a string") as exc:
        assert_historical_family_pairing(evil, HISTORICAL_FORAGER_FAMILY_ID)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "EvilStr" not in str(exc.value)


def test_hostile_right_without_repr_leak() -> None:
    evil = _EvilStr(HISTORICAL_FORAGER_FAMILY_ID)
    _EvilStr.calls = 0
    with pytest.raises(HistoricalForagerProvenanceError, match="must be a string"):
        assert_historical_family_pairing(HISTORICAL_FORAGER_FAMILY_ID, evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_string_subclass_rejected() -> None:
    with pytest.raises(HistoricalForagerProvenanceError, match="must be a string"):
        assert_historical_family_pairing(
            _StringSubclass(HISTORICAL_FORAGER_FAMILY_ID),
            HISTORICAL_FORAGER_FAMILY_ID,
        )  # type: ignore[arg-type]


def test_mismatch_sanitized_without_repr() -> None:
    with pytest.raises(
        HistoricalForagerFamilyMismatchError,
        match="historical reconstructed results pair only with",
    ) as exc:
        assert_historical_family_pairing("bad_left", "bad_right")
    assert "!r" not in str(exc.value)
    assert "bad_left" in str(exc.value)
    msg = str(exc.value)
    assert "'" in msg


def test_valid_pairing_passes() -> None:
    assert_historical_family_pairing(
        HISTORICAL_FORAGER_FAMILY_ID, HISTORICAL_FORAGER_FAMILY_ID
    )


def test_mismatch_even_one_bad_rejected_sanitized() -> None:
    with pytest.raises(HistoricalForagerFamilyMismatchError) as exc:
        assert_historical_family_pairing(HISTORICAL_FORAGER_FAMILY_ID, "other")
    assert "!r" not in str(exc.value)
