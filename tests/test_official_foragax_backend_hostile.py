"""Hostile string gate for official foragax qualification backend."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks import official_foragax as ofa

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def _dummy_args():
    return dict(
        executor={},
        entrypoints={},
        configuration={},
        run={"index": 0, "effective_seed": 0},
        invocation={"indices": [0]},
    )


def test_backend_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("cpu")
    _HostileStr.calls = 0
    args = _dummy_args()
    with pytest.raises(ofa.OfficialForagaxValidationError, match="backend is invalid"):
        ofa._qualification_workload_projection(backend=hostile, **args)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_backend_rejects_non_string_before_membership() -> None:
    args = _dummy_args()
    with pytest.raises(ofa.OfficialForagaxValidationError, match="backend is invalid"):
        ofa._qualification_workload_projection(backend=123, **args)  # type: ignore[arg-type]


def test_backend_rejects_unknown_benign() -> None:
    args = _dummy_args()
    with pytest.raises(ofa.OfficialForagaxValidationError, match="backend is invalid"):
        ofa._qualification_workload_projection(backend="unknown_backend_xyz", **args)


def test_benign_backend_passes_gate() -> None:
    args = _dummy_args()
    # valid backend should not raise backend error (may raise later about executor, but not backend)
    try:
        ofa._qualification_workload_projection(backend="cpu", **args)
    except ofa.OfficialForagaxValidationError as exc:
        assert "backend is invalid" not in str(exc)
    except Exception:
        pass
    try:
        ofa._qualification_workload_projection(backend="gpu", **args)
    except ofa.OfficialForagaxValidationError as exc:
        assert "backend is invalid" not in str(exc)
    except Exception:
        pass


