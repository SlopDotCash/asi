"""Hostile-safe validation for statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.utils.statistics import (
    bootstrap_ci,
    compute_statistics,
    ttest_comparison,
)


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


def test_compute_statistics_rejects_hostile_float_without_ratio() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        compute_statistics([1.0, 2.0], confidence_level=_HostileFloat(0.5))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_compute_statistics_rejects_string_subclass_confidence() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        compute_statistics([1.0, 2.0], confidence_level=_StringSubclass("0.95"))  # type: ignore[arg-type]


def test_compute_statistics_rejects_hostile_repr_confidence() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        compute_statistics([1.0, 2.0], confidence_level=_RaisingRepr())  # type: ignore[arg-type]


def test_compute_statistics_rejects_bool_confidence() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        compute_statistics([1.0, 2.0], confidence_level=True)  # type: ignore[arg-type]


def test_bootstrap_rejects_hostile_float_alpha() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_ci([1.0, 2.0, 3.0], confidence_level=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_bootstrap_rejects_string_subclass_statistic() -> None:
    with pytest.raises(ValueError, match="statistic"):
        bootstrap_ci([1.0, 2.0, 3.0], statistic=_StringSubclass("mean"))  # type: ignore[arg-type]


def test_bootstrap_rejects_hostile_repr_statistic() -> None:
    with pytest.raises(ValueError, match="statistic"):
        bootstrap_ci([1.0, 2.0, 3.0], statistic=_RaisingRepr())  # type: ignore[arg-type]


def test_bootstrap_rejects_hostile_int_n_bootstrap() -> None:
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_ci([1.0, 2.0, 3.0], n_bootstrap=_HostileInt(10))  # type: ignore[arg-type]


def test_ttest_rejects_hostile_float_alpha() -> None:
    a = [0.9, 0.8, 0.85]
    b = [0.5, 0.6, 0.55]
    with pytest.raises(ValueError, match="alpha"):
        ttest_comparison(a, b, alpha=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


@pytest.mark.parametrize("paired", [0, 1, np.bool_(True), "unpaired", None])
def test_ttest_requires_exact_paired_bool(paired: object) -> None:
    with pytest.raises(ValueError, match="paired must be an exact bool"):
        ttest_comparison(
            [101.0, 202.0, 303.0, 404.0],
            [100.0, 200.0, 300.0, 400.0],
            paired=paired,  # type: ignore[arg-type]
        )


def test_ttest_rejects_string_subclass_method_name() -> None:
    a = [0.9, 0.8, 0.85]
    b = [0.5, 0.6, 0.55]
    with pytest.raises(ValueError, match="method"):
        ttest_comparison(a, b, method_a=_StringSubclass("A"))  # type: ignore[arg-type]


def test_numpy_int_confidence_canonicalizes() -> None:
    # numpy scalars should be accepted and produce same result as float
    s_float = compute_statistics([1.0, 2.0, 3.0], confidence_level=0.95)
    s_np = compute_statistics([1.0, 2.0, 3.0], confidence_level=np.float64(0.95))
    assert s_float.ci_lower == pytest.approx(s_np.ci_lower)


def test_valid_confidence_and_alpha_pass() -> None:
    s = compute_statistics([1.0, 2.0, 3.0, 4.0], confidence_level=0.99)
    assert 0.0 < s.mean < 10.0
    res = ttest_comparison([0.9, 0.8, 0.85], [0.5, 0.6, 0.55], alpha=0.05)
    assert 0.0 <= res.p_value <= 1.0
