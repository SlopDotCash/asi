"""Hostile string gate for streaming summary environment before hash."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import (
    ENVIRONMENT_ROSTER,
    StreamingRunSummary,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")


def _legal_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "environment_kind": ENVIRONMENT_ROSTER[0],
        "horizon": 10,
        "phase_length": 2,
        "n_states": 2,
        "early_window": 1,
        "late_window": 1,
        "post_switch_window": 1,
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def test_streaming_summary_rejects_hostile_env_before_hash() -> None:
    hostile = _HostileStr(ENVIRONMENT_ROSTER[0])
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="unsupported streaming-summary environment"):
        StreamingRunSummary(**_legal_kwargs(environment_kind=hostile))  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    # also test unknown env still raises without hostile
    with pytest.raises(ValueError, match="unsupported streaming-summary environment"):
        StreamingRunSummary(**_legal_kwargs(environment_kind="unknown_env"))  # type: ignore[arg-type]
    # benign still works
    summary = StreamingRunSummary(**_legal_kwargs())  # type: ignore[arg-type]
    assert summary is not None


def test_streaming_summary_rejects_non_str_env() -> None:
    with pytest.raises(ValueError, match="unsupported streaming-summary environment"):
        StreamingRunSummary(**_legal_kwargs(environment_kind=True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported streaming-summary environment"):
        StreamingRunSummary(**_legal_kwargs(environment_kind=123))  # type: ignore[arg-type]


def test_streaming_summary_hostile_not_in_repr() -> None:
    hostile = _HostileStr("evil_env")
    _HostileStr.calls = 0
    try:
        StreamingRunSummary(**_legal_kwargs(environment_kind=hostile))  # type: ignore[arg-type]
    except ValueError as exc:
        assert "_HostileStr" not in str(exc)
        assert "evil_env" not in str(exc)
        assert _HostileStr.calls == 0
    else:
        raise AssertionError("should have raised")
