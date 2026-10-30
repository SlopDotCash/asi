"""Hostile int identity gates for the recurring IPMNIST numeric helpers.

The remaining artifact numeric gates were hardened to reject hostile ``int``
and ``float`` subclasses before any trusted numeric conversion (#1944 and the
follow-up that added ``tests/test_artifact_numeric_gates_hostile.py``).  The
recurring IPMNIST screening helpers still used ``isinstance``, so a hostile
``int`` subclass passed the gate and its overridden dunder ran during the
positive-value comparison and the later ``int()`` conversion.

Each gate below must reject the hostile value and must not execute any of its
dunders.  ``calls`` staying at zero is the observable contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    _validated_recurring_phase_lengths,
    build_recurring_ipmnist_online_indices,
    run_recurring_ipmnist_retention_development,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile le must not run")

    def __int__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile int must not run")

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile index must not run")

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")


def _reset() -> None:
    _HostileInt.calls = 0


DATA_X = np.asarray(
    [
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 0.5, 0.0, -0.5],
        [0.0, 1.0, -1.0, 0.5],
        [0.5, -1.0, 1.0, 0.0],
        [-0.5, 0.0, 0.5, 1.0],
        [0.25, -0.25, 0.75, -0.75],
        [-0.75, 0.75, -0.25, 0.25],
        [0.1, 0.2, 0.3, 0.4],
        [-0.4, -0.3, -0.2, -0.1],
    ],
    dtype=np.float32,
)
DATA_Y = np.asarray([0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
PERMUTATION_A = np.asarray([0, 1, 2, 3], dtype=np.int32)
PERMUTATION_B = np.asarray([3, 1, 0, 2], dtype=np.int32)
SENTINELS = (7, 8)
CONFIG = IPMNISTConfig(
    n_tasks=3,
    task_length=2,
    input_dim=4,
    hidden1=3,
    hidden2=2,
    n_classes=2,
)


def test_phase_length_gate_rejects_hostile_int_before_comparison() -> None:
    _reset()
    with pytest.raises(ValueError):
        _validated_recurring_phase_lengths(
            (_HostileInt(2), 3, 2)  # type: ignore[arg-type]
        )
    assert _HostileInt.calls == 0


def test_n_examples_gate_rejects_hostile_int_before_comparison() -> None:
    _reset()
    with pytest.raises(ValueError):
        build_recurring_ipmnist_online_indices(
            seed=19,
            n_examples=_HostileInt(9),  # type: ignore[arg-type]
            phase_lengths=(2, 3, 2),
            sentinel_indices=SENTINELS,
        )
    assert _HostileInt.calls == 0


def test_relearning_window_gate_rejects_hostile_int_before_comparison() -> None:
    _reset()
    with pytest.raises(ValueError):
        run_recurring_ipmnist_retention_development(
            DATA_X,
            DATA_Y,
            screening_spec("sgd_ema_norm"),
            seed=19,
            config=CONFIG,
            phase_lengths=(2, 3, 2),
            permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A.copy()),
            sentinel_indices=SENTINELS,
            relearning_window=_HostileInt(1),  # type: ignore[arg-type]
        )
    assert _HostileInt.calls == 0


def test_gates_accept_benign_builtin_ints() -> None:
    """The exact-type gates must still admit ordinary built-in integers."""

    _reset()
    assert _validated_recurring_phase_lengths((2, 3, 2)) == (2, 3, 2)

    a_order, b_order, recurring_a_order = build_recurring_ipmnist_online_indices(
        seed=19,
        n_examples=9,
        phase_lengths=(2, 3, 2),
        sentinel_indices=SENTINELS,
    )
    assert a_order.shape == (2,)
    assert b_order.shape == (3,)
    assert np.array_equal(a_order, recurring_a_order)

    report = run_recurring_ipmnist_retention_development(
        DATA_X,
        DATA_Y,
        screening_spec("sgd_ema_norm"),
        seed=19,
        config=CONFIG,
        phase_lengths=(2, 3, 2),
        permutations=(PERMUTATION_A, PERMUTATION_B, PERMUTATION_A.copy()),
        sentinel_indices=SENTINELS,
        relearning_window=1,
    )
    assert report.recurrence.relearning_window == 1
    assert _HostileInt.calls == 0


@pytest.mark.parametrize("bad", [True, False, 1.0, "2", None])
def test_phase_length_gate_still_rejects_non_ints(bad: object) -> None:
    _reset()
    with pytest.raises(ValueError):
        _validated_recurring_phase_lengths(
            (bad, 3, 2)  # type: ignore[list-item]
        )
    assert _HostileInt.calls == 0
