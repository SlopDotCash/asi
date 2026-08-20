"""Scale-robust pair records enforce their collection and identity bounds."""

from __future__ import annotations

from typing import Any, cast

import pytest

from alberta_framework.evaluation.scale_robust_feature import (
    _MAX_INTEGER_PAIRS,
    CONDITION_PRIMARY,
    ConditionSeedRecord,
    count_relevant_context_pairs,
    count_relevant_context_pairs_by_task,
)

pytestmark = pytest.mark.unit

_PAIR = (0, 12)


class _HostileMeta(type):
    calls = 0

    def __eq__(cls, other: object) -> bool:
        del other
        cls.calls += 1
        raise AssertionError("hostile metaclass equality must not run")

    def __hash__(cls) -> int:
        cls.calls += 1
        raise AssertionError("hostile metaclass hashing must not run")


class _HostileSequence(tuple[object, ...], metaclass=_HostileMeta):
    pass


def test_frozen_pair_count_bound() -> None:
    assert _MAX_INTEGER_PAIRS == 4096


def test_last_fit_pair_count_is_accepted() -> None:
    pairs = (_PAIR,) * _MAX_INTEGER_PAIRS
    record = ConditionSeedRecord(
        seed=0,
        condition=CONDITION_PRIMARY,
        phases=(),
        end_segment_5_active_pairs=pairs,
        end_segment_7_active_pairs=(),
        final_active_pairs=(),
    )
    assert len(record.end_segment_5_active_pairs) == _MAX_INTEGER_PAIRS
    assert count_relevant_context_pairs(pairs) == 1
    assert count_relevant_context_pairs_by_task(pairs) == (1, 0)


def test_first_oversized_pair_collection_is_rejected_by_all_consumers() -> None:
    pairs = (_PAIR,) * (_MAX_INTEGER_PAIRS + 1)
    with pytest.raises(ValueError, match="4096-pair limit"):
        ConditionSeedRecord(
            seed=0,
            condition=CONDITION_PRIMARY,
            phases=(),
            end_segment_5_active_pairs=pairs,
            end_segment_7_active_pairs=(),
            final_active_pairs=(),
        )
    with pytest.raises(ValueError, match="4096-pair limit"):
        count_relevant_context_pairs(pairs)
    with pytest.raises(ValueError, match="4096-pair limit"):
        count_relevant_context_pairs_by_task(pairs)


def test_pair_admission_rejects_hostile_sequence_without_type_hooks() -> None:
    _HostileMeta.calls = 0
    hostile = _HostileSequence((_PAIR,))
    hostile_pairs = cast(Any, hostile)
    with pytest.raises(TypeError, match="exact tuple or list"):
        count_relevant_context_pairs(hostile_pairs)
    with pytest.raises(TypeError, match="exact tuple or list"):
        count_relevant_context_pairs_by_task(hostile_pairs)
    assert _HostileMeta.calls == 0


def test_pair_counters_reject_hostile_nested_values_before_comparison() -> None:
    class HostileInt(int):
        def __lt__(self, other: object) -> bool:
            del other
            raise AssertionError("hostile comparison must not run")

        def __hash__(self) -> int:
            raise AssertionError("hostile hashing must not run")

    pairs = ((HostileInt(0), 12),)
    with pytest.raises(ValueError, match="must contain integers"):
        count_relevant_context_pairs(pairs)
    with pytest.raises(ValueError, match="must contain integers"):
        count_relevant_context_pairs_by_task(pairs)
