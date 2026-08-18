"""Leftover-identity gates for publication significance records."""

import pickle

import pytest

from alberta_framework.utils.statistics import SignificanceResult


def _legal() -> SignificanceResult:
    return SignificanceResult("paired", 1.0, 0.01, True, 0.05, 0.2, "a", "b")


@pytest.mark.parametrize("significant", [1, 0, "FIXED"])
def test_significance_result_rejects_leftover_bool_identities(significant: object) -> None:
    with pytest.raises(ValueError, match="significant"):
        SignificanceResult(  # type: ignore[arg-type]
            "paired", 1.0, 0.01, significant, 0.05, 0.2, "a", "b"
        )


@pytest.mark.parametrize(
    ("position", "value", "message"),
    [(0, True, "test_name"), (6, 1, "method_a"), (7, False, "method_b")],
)
def test_significance_result_rejects_leftover_name_identities(
    position: int, value: object, message: str
) -> None:
    fields: list[object] = ["paired", 1.0, 0.01, True, 0.05, 0.2, "a", "b"]
    fields[position] = value
    with pytest.raises(ValueError, match=message):
        SignificanceResult(*fields)  # type: ignore[arg-type]


def test_significance_result_preserves_namedtuple_contract() -> None:
    result = _legal()
    assert isinstance(result, tuple)
    assert tuple(result) == ("paired", 1.0, 0.01, True, 0.05, 0.2, "a", "b")
    assert result._fields == (
        "test_name",
        "statistic",
        "p_value",
        "significant",
        "alpha",
        "effect_size",
        "method_a",
        "method_b",
    )
    assert result._asdict()["significant"] is True
    assert result._replace(significant=False).significant is False
    restored = pickle.loads(pickle.dumps(result))
    assert isinstance(restored, SignificanceResult)
    assert restored == result
