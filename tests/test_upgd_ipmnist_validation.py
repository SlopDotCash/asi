"""Committed regressions for IPMNISTConfig fail-closed validation."""

import numpy as np
import pytest

from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

_INT32_MAX = 2**31 - 1


def _valid_config(**overrides) -> IPMNISTConfig:
    base = dict(n_tasks=2, task_length=5, input_dim=8, hidden1=4, hidden2=4, n_classes=2)
    base.update(overrides)
    return IPMNISTConfig(**base)


def test_accepts_numpy_int_types():
    for typ in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ):
        cfg = _valid_config(n_tasks=typ(2))
        assert cfg.n_tasks == 2


def test_rejects_bool_and_np_bool():
    with pytest.raises(ValueError):
        _valid_config(n_tasks=True)
    with pytest.raises(ValueError):
        _valid_config(n_tasks=np.bool_(True))


def test_rejects_float_and_str():
    with pytest.raises(ValueError):
        _valid_config(n_tasks=2.0)
    with pytest.raises(ValueError):
        _valid_config(n_tasks="2")
    with pytest.raises(ValueError):
        _valid_config(n_tasks=float("nan"))
    with pytest.raises(ValueError):
        _valid_config(n_tasks=float("inf"))


def test_rejects_hostile_subclass():
    class EvilInt(int):
        def __repr__(self):
            raise AssertionError("hostile repr must not be invoked")

    with pytest.raises(ValueError) as exc:
        _valid_config(n_tasks=EvilInt(2))
    # must not have invoked hostile repr
    assert "EvilInt" not in str(exc.value)


def test_hostile_repr_not_interpolated():
    class Hostile:
        def __index__(self):
            return 2

        def __repr__(self):
            raise AssertionError("repr escape")

    # Hostile object whose type is not in allowlist should fail before repr
    with pytest.raises(ValueError):
        _valid_config(n_tasks=Hostile())  # type: ignore[arg-type]


def test_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        _valid_config(n_tasks=0)
    with pytest.raises(ValueError):
        _valid_config(n_tasks=-1)
    with pytest.raises(ValueError):
        _valid_config(n_tasks=_INT32_MAX + 1)
    with pytest.raises(ValueError):
        _valid_config(n_tasks=2**60)


def test_derived_n_tasks_task_length_overflow():
    # n_tasks * task_length must be <= INT32_MAX
    with pytest.raises(ValueError, match="n_tasks \\* task_length"):
        _valid_config(n_tasks=50000, task_length=50000)  # product 2.5e9 > INT32_MAX


def test_derived_n_tasks_input_dim_overflow():
    with pytest.raises(ValueError, match="n_tasks \\* input_dim"):
        _valid_config(n_tasks=50000, input_dim=50000)


def test_derived_input_hidden_overflow():
    with pytest.raises(ValueError, match="input_dim \\* hidden1"):
        _valid_config(input_dim=50000, hidden1=50000)


def test_derived_hidden_hidden_overflow():
    with pytest.raises(ValueError, match="hidden1 \\* hidden2"):
        _valid_config(hidden1=50000, hidden2=50000)


def test_derived_hidden_n_classes_overflow():
    with pytest.raises(ValueError, match="hidden2 \\* n_classes"):
        _valid_config(hidden2=50000, n_classes=50000)


def test_valid_derived_within_bounds():
    cfg = _valid_config(
        n_tasks=200, task_length=5000, input_dim=784, hidden1=300, hidden2=150,
        n_classes=10,
    )
    assert cfg.n_tasks == 200
    assert cfg.n_steps == 200 * 5000
