"""Hostile validation for rule discovery search-resource gates.

Protocol-bound per-name caps plus combined work-unit products must reject
before JAX allocation, stream construction, or range loops. Trusted array
identity is exact-gated before shape or conversion hooks. The exact-type
hostile gate stays; 2**31-1 is not a legal search int.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from alberta_framework.benchmarks.micro_continual import MICRO_SUITE
from alberta_framework.benchmarks.rule_discovery import (
    _DEFAULT_BASELINE_EVALS,
    _DEFAULT_DIGITS_HOLDOUT_STEPS,
    _DEFAULT_DIGITS_SEARCH_STEPS,
    _DEFAULT_EVAL_SEEDS,
    _DEFAULT_GAUSS_STEPS,
    _DEFAULT_HOLDOUT_EVALS,
    _DEFAULT_HOLDOUT_SEEDS,
    _DEFAULT_SEARCH_EVALS,
    _SEARCH_CANDIDATE_EVALS_MAX,
    _SEARCH_INT_MAX_BY_NAME,
    _SEARCH_LOGICAL_WORK_MAX,
    _SEARCH_STREAM_STEPS_MAX,
    GENOME_SIZE,
    _require_search_int,
    _require_search_protocol_work,
    _require_search_work_unit,
    _require_suite_streams,
    _resolved_suite,
    evaluate_population,
)


class _HostileInt(int):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("repr hook")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("str hook")


class _EvilStr(str):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("str hook")


class _HostileGenomes:
    calls = 0

    @property
    def __class__(self) -> type[np.ndarray]:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("class hook")

    @property
    def shape(self) -> tuple[int, ...]:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("shape hook")

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:  # pragma: no cover
        del args, kwargs
        type(self).calls += 1
        raise AssertionError("array conversion hook")

    def __jax_array__(self) -> np.ndarray:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("jax conversion hook")


def test_require_search_int_rejects_hostile_without_hooks() -> None:
    evil = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_search_int("my_param", evil, minimum=1)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "!r" not in str(exc.value)
    assert "HostileInt" not in str(exc.value)


def test_require_search_int_rejects_string_subclass_name() -> None:
    evil = _EvilStr("my_param")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _require_search_int(evil, 1, minimum=0)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_require_search_int_rejects_below_minimum_sanitized() -> None:
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_search_int("my_param", 0, minimum=1)
    assert "!r" not in str(exc.value)
    assert "my_param" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/rule_discovery.py").read_text()
    assert "!r" not in text


def test_valid_int_passes() -> None:
    assert _require_search_int("my_param", 5, minimum=1) == 5


def test_require_search_int_rejects_unbounded_and_int32_max() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _require_search_int("task_length", 10**12, minimum=1)
    with pytest.raises(ValueError, match="must be an integer"):
        _require_search_int("n_random", 2**31 - 1, minimum=0)
    with pytest.raises(ValueError, match="must be an integer"):
        _require_search_int("generations", 2**31 - 1, minimum=0)


def test_require_search_int_last_fit_and_first_overflow() -> None:
    for name, maximum in _SEARCH_INT_MAX_BY_NAME.items():
        assert _require_search_int(name, maximum, minimum=0) == maximum
        with pytest.raises(ValueError, match="must be an integer"):
            _require_search_int(name, maximum + 1, minimum=0)


def test_require_search_work_unit_combined_product() -> None:
    last_fit_rows = _SEARCH_INT_MAX_BY_NAME["n_random"]
    _require_search_work_unit(n_random=last_fit_rows)
    with pytest.raises(ValueError, match="candidate arrays"):
        _require_search_work_unit(n_random=last_fit_rows + 1)
    _require_search_work_unit(
        n_random=0, n_tasks=2, task_length=_SEARCH_STREAM_STEPS_MAX // 2
    )
    with pytest.raises(ValueError, match="stream steps"):
        _require_search_work_unit(n_random=0, n_tasks=2, task_length=5_000_001)


def test_require_search_work_unit_adjacent_population_generations() -> None:
    _require_search_work_unit(n_random=0, population=1_024, generations=15)
    with pytest.raises(ValueError, match="candidate evaluations"):
        _require_search_work_unit(n_random=0, population=1_024, generations=16)


def test_require_search_work_unit_adjacent_children_generations() -> None:
    _require_search_work_unit(n_random=4_096, children=256, generations=48)
    with pytest.raises(ValueError, match="candidate evaluations"):
        _require_search_work_unit(n_random=4_097, children=256, generations=48)


def test_resolved_suite_rejects_single_override_stream_product() -> None:
    last_fit_length = _SEARCH_STREAM_STEPS_MAX // 12
    _resolved_suite(None, last_fit_length)
    with pytest.raises(ValueError, match="stream steps"):
        _resolved_suite(None, last_fit_length + 1)


def test_evaluate_population_rejects_hostile_genomes_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("hostile genomes reached stream materialization")

    monkeypatch.setattr(
        "alberta_framework.benchmarks.rule_discovery._materialize_eval",
        unexpected_materialization,
    )
    _HostileGenomes.calls = 0
    with pytest.raises(TypeError, match="trusted array"):
        evaluate_population(
            _HostileGenomes(),  # type: ignore[arg-type]
            MICRO_SUITE["M1"],
            seeds=(0,),
            batch_size=1,
        )
    assert _HostileGenomes.calls == 0


def test_evaluate_population_rejects_oversize_block_before_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("oversize genome block reached stream materialization")

    monkeypatch.setattr(
        "alberta_framework.benchmarks.rule_discovery._materialize_eval",
        unexpected_materialization,
    )
    genomes = np.zeros((_SEARCH_INT_MAX_BY_NAME["n_random"] + 1, GENOME_SIZE), np.float32)
    with pytest.raises(ValueError, match="must be an integer"):
        evaluate_population(
            genomes,
            MICRO_SUITE["M1"],
            seeds=(0,),
            batch_size=1,
        )


def test_require_search_work_unit_cartesian_last_fit_and_first_overflow() -> None:
    evals = _SEARCH_CANDIDATE_EVALS_MAX
    last_fit_steps = _SEARCH_LOGICAL_WORK_MAX // evals
    assert last_fit_steps <= _SEARCH_STREAM_STEPS_MAX
    _require_search_work_unit(
        n_random=0,
        population=1_024,
        generations=15,
        n_tasks=1,
        task_length=last_fit_steps,
        n_seeds=1,
    )
    with pytest.raises(ValueError, match="logical work"):
        _require_search_work_unit(
            n_random=0,
            population=1_024,
            generations=15,
            n_tasks=1,
            task_length=last_fit_steps + 1,
            n_seeds=1,
        )


def test_require_search_work_unit_rejects_int32_envelope_product() -> None:
    with pytest.raises(ValueError, match="logical work"):
        _require_search_work_unit(
            n_random=0,
            population=1_024,
            generations=15,
            n_tasks=1,
            task_length=_SEARCH_STREAM_STEPS_MAX,
            n_seeds=1,
        )


def test_require_search_protocol_work_default_digits_and_gauss_fit() -> None:
    _require_search_protocol_work(
        search_evals=_DEFAULT_SEARCH_EVALS,
        search_steps=_DEFAULT_DIGITS_SEARCH_STEPS,
        n_eval_seeds=_DEFAULT_EVAL_SEEDS,
        baseline_evals=_DEFAULT_BASELINE_EVALS,
        holdout_evals=_DEFAULT_HOLDOUT_EVALS,
        holdout_steps=_DEFAULT_DIGITS_HOLDOUT_STEPS,
        n_holdout_seeds=_DEFAULT_HOLDOUT_SEEDS,
    )
    _require_search_protocol_work(
        search_evals=_DEFAULT_SEARCH_EVALS,
        search_steps=_DEFAULT_GAUSS_STEPS,
        n_eval_seeds=_DEFAULT_EVAL_SEEDS,
        baseline_evals=_DEFAULT_BASELINE_EVALS,
        holdout_evals=_DEFAULT_HOLDOUT_EVALS,
        holdout_steps=_DEFAULT_GAUSS_STEPS,
        n_holdout_seeds=_DEFAULT_HOLDOUT_SEEDS,
    )


def test_require_suite_streams_bounds_selected_product_not_each_factor() -> None:
    left = dataclasses.replace(MICRO_SUITE["M1"], n_tasks=1, task_length=5_000_000)
    right = dataclasses.replace(MICRO_SUITE["M2"], n_tasks=1, task_length=5_000_000)
    with pytest.raises(ValueError, match="logical work"):
        _require_suite_streams(
            {"M1": left, "M2": right},
            selected=("M1", "M2"),
            n_seeds=1,
            n_random=0,
            population=1_024,
            generations=15,
        )


def test_evaluate_population_rejects_cartesian_work_before_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("cartesian overflow reached stream materialization")

    monkeypatch.setattr(
        "alberta_framework.benchmarks.rule_discovery._materialize_eval",
        unexpected_materialization,
    )
    n_genomes = 2_048
    last_fit = _SEARCH_LOGICAL_WORK_MAX // n_genomes
    genomes = np.zeros((n_genomes, GENOME_SIZE), np.float32)
    overflow = dataclasses.replace(
        MICRO_SUITE["M1"], n_tasks=1, task_length=last_fit + 1
    )
    with pytest.raises(ValueError, match="logical work"):
        evaluate_population(genomes, overflow, seeds=(0,), batch_size=1)
