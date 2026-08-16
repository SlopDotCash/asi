"""Unit tests for the Step 1 integer trust boundary (issue #499).

The Step 1 facade must admit only exact builtin/NumPy integer scalar types.
An actual ``int`` subclass can override ``__int__`` (or ``__repr__``) with
hostile hooks, so subclasses are rejected by identity comparison before any
conversion runs, and the invalid-type error never interpolates the value.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.steps import Step1KernelConfig, run_step1_smoke

pytestmark = pytest.mark.unit

_INT_FIELDS = ("steps", "final_window")
_CONFIG_INT_FIELDS = ("feature_dim", "num_relevant")


class _LyingIntSubclass(int):
    """An actual ``int`` subclass whose conversion hook rewrites the value."""

    def __int__(self) -> int:
        return 2

    def __index__(self) -> int:
        return 2


class _RaisingIntSubclass(int):
    """An actual ``int`` subclass whose conversion hook raises."""

    def __int__(self) -> int:
        raise RuntimeError("conversion hook must not run")

    def __index__(self) -> int:
        raise RuntimeError("conversion hook must not run")


class _RaisingRepr:
    """A non-integer whose ``__repr__`` hook raises if the error path runs it."""

    def __repr__(self) -> str:
        raise RuntimeError("repr hook must not run")


def _smoke_kwargs(field: str, value: object) -> dict[str, object]:
    kwargs: dict[str, object] = {"steps": 4, "final_window": 2}
    kwargs[field] = value
    return kwargs


@pytest.mark.parametrize("field", _INT_FIELDS)
def test_step1_smoke_rejects_int_subclass_with_lying_conversion_hook(field: str) -> None:
    value = _LyingIntSubclass(-1)
    assert isinstance(value, int)

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        run_step1_smoke(**_smoke_kwargs(field, value))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _INT_FIELDS)
def test_step1_smoke_rejects_int_subclass_without_running_raising_hook(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        run_step1_smoke(**_smoke_kwargs(field, _RaisingIntSubclass(4)))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _INT_FIELDS)
def test_step1_smoke_invalid_type_error_does_not_run_repr_hook(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        run_step1_smoke(**_smoke_kwargs(field, _RaisingRepr()))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _CONFIG_INT_FIELDS)
def test_step1_config_rejects_int_subclass_with_lying_conversion_hook(field: str) -> None:
    value = _LyingIntSubclass(-1)

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        Step1KernelConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _CONFIG_INT_FIELDS)
def test_step1_config_rejects_int_subclass_without_running_raising_hook(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        Step1KernelConfig(**{field: _RaisingIntSubclass(4)})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _CONFIG_INT_FIELDS)
def test_step1_config_invalid_type_error_does_not_run_repr_hook(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        Step1KernelConfig(**{field: _RaisingRepr()})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "np_type",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ],
)
def test_step1_config_canonicalizes_exact_numpy_integer_scalars(np_type: type) -> None:
    config = Step1KernelConfig(
        feature_dim=np_type(10),  # type: ignore[arg-type]
        num_relevant=np_type(3),  # type: ignore[arg-type]
    )

    assert type(config.feature_dim) is int
    assert type(config.num_relevant) is int
    assert config.feature_dim == 10
    assert config.num_relevant == 3


@pytest.mark.parametrize("field", _CONFIG_INT_FIELDS)
@pytest.mark.parametrize("value", [True, np.bool_(True), 4.0, np.float64(4.0), "4", None])
def test_step1_config_still_rejects_non_integer_scalars(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        Step1KernelConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _INT_FIELDS)
def test_step1_smoke_bounds_errors_still_name_the_field(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        run_step1_smoke(**_smoke_kwargs(field, 0))  # type: ignore[arg-type]
