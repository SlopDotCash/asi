"""Hostile validation for utils/experiments facade."""

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from alberta_framework.utils.experiments import (
    AggregatedResults,
    MetricSummary,
    _require_finite_metric_array,
    _require_hyperparameter_coordinate,
    extract_hyperparameter_results,
    get_final_performance,
)


def _aggregated(name: str, n_seeds: int = 2, n_steps: int = 10) -> AggregatedResults:
    arr = np.ones((n_seeds, n_steps), dtype=np.float64)
    summary = MetricSummary(mean=1.0, std=0.0, min=1.0, max=1.0, n_seeds=n_seeds, values=arr[:, -1])
    return AggregatedResults(
        config_name=name,
        seeds=list(range(n_seeds)),
        metric_arrays={"squared_error": arr},
        summary={"squared_error": summary},
    )


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.as_integer_ratio must not be called")

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.__float__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileFloat.__repr__ must not be called")


class _HostileHash:
    calls = 0

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileHash.__hash__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileHash.__repr__ must not be called")


def test_rejects_string_subclass_for_metric() -> None:
    arr = np.ones((2, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_finite_metric_array(arr, _StringSubclass("squared_error"))  # type: ignore[arg-type]


def test_hostile_str_for_metric_without_repr_leak() -> None:
    arr = np.ones((2, 5), dtype=np.float64)
    evil = _EvilStr("squared_error")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_finite_metric_array(arr, evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int_for_window() -> None:
    results = {"a": _aggregated("a")}
    with pytest.raises(ValueError, match="must be a positive built-in integer"):
        get_final_performance(results, window=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a positive built-in integer") as exc:
        get_final_performance(results, window=_HostileInt(10))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_metric_via_get_final_without_repr() -> None:
    results = {"a": _aggregated("a")}
    evil = _EvilStr("squared_error")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        get_final_performance(results, metric=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_does_not_invoke_hostile_value_when_name_is_evil_via_coordinate() -> None:
    evil = _EvilStr("x")
    hostile = _HostileHash()
    _HostileHash.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_hyperparameter_coordinate(hostile, name=evil)  # type: ignore[arg-type]
    assert _HostileHash.calls == 0


def test_rejects_string_subclass_for_coordinate_name() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_hyperparameter_coordinate(1, name=_StringSubclass("a"))  # type: ignore[arg-type]


def test_rejects_noncanonical_coordinate_without_repr() -> None:
    with pytest.raises(ValueError, match="noncanonical coordinate") as exc:
        _require_hyperparameter_coordinate([], name="a")
    assert "!r" not in str(exc.value)
    assert "a" in str(exc.value)  # sanitized with quotes but contains name


def test_rejects_duplicate_names_without_repr() -> None:
    from alberta_framework.core.learners import LinearLearner  # noqa: F401
    from alberta_framework.streams.base import ScanStream  # noqa: F401
    from alberta_framework.utils.experiments import ExperimentConfig

    def _learner():  # type: ignore[no-untyped-def]
        raise AssertionError("should not be called")

    def _stream():  # type: ignore[no-untyped-def]
        raise AssertionError("should not be called")

    cfg1 = ExperimentConfig(
        name="dup",
        learner_factory=_learner,
        stream_factory=_stream,
        num_steps=1,
    )
    cfg2 = ExperimentConfig(
        name="dup",
        learner_factory=_learner,
        stream_factory=_stream,
        num_steps=1,
    )
    from alberta_framework.utils.experiments import run_multi_seed_experiment

    with pytest.raises(ValueError, match="must be unique") as exc:
        run_multi_seed_experiment([cfg1, cfg2], seeds=[0], parallel=False)
    assert "!r" not in str(exc.value)
    assert "dup" in str(exc.value)


def test_rejects_incompatible_families_without_repr() -> None:
    results = {
        "a": _aggregated("a"),
        "b": _aggregated("b"),
    }
    # python int vs numpy int32 are incompatible families
    def _extractor(name: str) -> Any:
        return 1 if name == "a" else np.int32(1)

    with pytest.raises(ValueError, match="mutually compatible") as exc:
        extract_hyperparameter_results(results, param_extractor=_extractor)
    assert "!r" not in str(exc.value)
    assert "a" in str(exc.value) or "b" in str(exc.value)


def test_rejects_colliding_coordinates_without_repr() -> None:
    results = {
        "a": _aggregated("a"),
        "b": _aggregated("b"),
    }
    with pytest.raises(ValueError, match="maps several configurations") as exc:
        extract_hyperparameter_results(results, param_extractor=lambda _: 1)
    assert "!r" not in str(exc.value)
    assert "a" in str(exc.value)


def test_valid_configs_still_pass() -> None:
    results = {"a": _aggregated("a")}
    perf = get_final_performance(results, window=5)
    assert "a" in perf
    # canonical coordinates
    coord = _require_hyperparameter_coordinate(1, name="a")
    assert coord == 1
    coord2 = _require_hyperparameter_coordinate("hello", name="a")
    assert coord2 == "hello"
    coord3 = _require_hyperparameter_coordinate((1, "x"), name="a")
    assert coord3 == (1, "x")


def test_numpy_scalars_pass() -> None:
    # metric arrays with numpy finite values
    arr = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    _require_finite_metric_array(arr, "squared_error")
    # canonical numpy scalar coordinate
    c = _require_hyperparameter_coordinate(np.int32(5), name="a")
    assert int(c) == 5
    # Fraction canonical
    c2 = _require_hyperparameter_coordinate(Fraction(1, 2), name="a")
    assert c2 == Fraction(1, 2)


def test_hostile_coordinate_is_rejected_without_hook() -> None:
    # HostileFloat subclass should be rejected as noncanonical without calling hooks
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="noncanonical coordinate") as exc:
        _require_hyperparameter_coordinate(_HostileFloat(0.1), name="a")
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)
