"""Reject unbounded rule-discovery search ints before genome/stream allocation."""

from __future__ import annotations

import jax.random as jr
import pytest

import alberta_framework.benchmarks.rule_discovery as rule_discovery
from alberta_framework.benchmarks.micro_continual import MICRO_SUITE
from alberta_framework.benchmarks.rule_discovery import (
    _require_search_int,
    run_search,
    tune_champion_baseline,
)

pytestmark = pytest.mark.unit

_OVERFLOW_SEARCH_INT = 10**12
_SEARCH_INT_LIMIT = 1_000_000


def test_require_search_int_rejects_trillion_n_random() -> None:
    with pytest.raises(ValueError, match="n_random"):
        _require_search_int("n_random", _OVERFLOW_SEARCH_INT, minimum=0)


def test_require_search_int_rejects_first_overflow_int() -> None:
    with pytest.raises(ValueError, match="search integer limit"):
        _require_search_int("generations", _SEARCH_INT_LIMIT + 1, minimum=0)


def test_require_search_int_last_fit_still_accepted() -> None:
    assert _require_search_int("n_random", _SEARCH_INT_LIMIT, minimum=0) == _SEARCH_INT_LIMIT
    assert _require_search_int("n_random", 0, minimum=0) == 0
    assert _require_search_int("population", 3072, minimum=1) == 3072


def test_run_search_rejects_trillion_n_random_before_genome_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_generation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("overflow n_random reached genome generation")

    monkeypatch.setattr(rule_discovery, "seed_genomes", unexpected_generation)
    monkeypatch.setattr(rule_discovery, "random_genomes", unexpected_generation)
    with pytest.raises(ValueError, match="n_random"):
        run_search(
            n_random=_OVERFLOW_SEARCH_INT,
            population=2,
            generations=0,
            elite=1,
            eval_seeds=(0,),
            holdout_seeds=(101,),
            top_k=1,
            batch_size=2,
        )


def test_resolved_suite_rejects_trillion_task_length() -> None:
    with pytest.raises(ValueError, match="task_length"):
        rule_discovery._resolved_suite(None, _OVERFLOW_SEARCH_INT)


def test_tune_champion_baseline_rejects_trillion_generations_before_genomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_generation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("overflow generations reached genome generation")

    monkeypatch.setattr(rule_discovery, "random_genomes", unexpected_generation)
    with pytest.raises(ValueError, match="generations"):
        tune_champion_baseline(
            jr.key(0),
            task_names=("M1",),
            eval_seeds=(0,),
            batch_size=2,
            suite=MICRO_SUITE,
            n_random=2,
            generations=_OVERFLOW_SEARCH_INT,
            children=1,
        )
