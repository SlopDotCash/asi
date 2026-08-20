"""CORA arm receipts reject oversized evaluation matrices before splat hang.

Origin ``ArmResult`` walked every checkpoint row, then splatted every cell into
a tuple, before comparing length to the frozen 7-row analogue. A cheap
``(row,) * 8_000_000`` repeated pointer tuple took 3.240s on origin/main.
"""

from __future__ import annotations

import time

import pytest

from alberta_framework.benchmarks.cora_development import (
    _EVALUATION_MATRIX_ROWS,
    TASK_TARGETS,
    ArmResult,
    ResourceReceipt,
    _metrics,
)


def _receipt() -> ResourceReceipt:
    return ResourceReceipt(
        training_environment_steps=1,
        evaluation_environment_steps=1,
        model_queries=1,
        agent_updates=1,
        replay_inserts=1,
        replay_samples=1,
        persistent_bytes=1,
        peak_replay_bytes=1,
        logical_compute_units=1,
        elapsed_ns=1,
    )


def _row() -> tuple[float, ...]:
    return tuple(0.0 for _ in TASK_TARGETS)


def _matrix(rows: int) -> tuple[tuple[float, ...], ...]:
    return (_row(),) * rows


def test_frozen_checkpoint_count_is_the_analogue_row_bound() -> None:
    assert _EVALUATION_MATRIX_ROWS == 7


def test_last_fit_checkpoint_count_is_accepted() -> None:
    result = ArmResult(
        arm_id="replay_q",
        training_return=0.0,
        evaluation_matrix=_matrix(_EVALUATION_MATRIX_ROWS),
        continual_evaluation=0.0,
        isolated_forgetting=0.0,
        isolated_forward_transfer=0.0,
        receipt=_receipt(),
        candidate_eligible=True,
    )
    assert len(result.evaluation_matrix) == _EVALUATION_MATRIX_ROWS
    assert _metrics(result.evaluation_matrix)[0] == 0.0


@pytest.mark.parametrize("rows", [1, 6, 8, 8_000_000])
def test_oversized_or_short_matrix_rejects_before_cell_splat(rows: int) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="checkpoint rows"):
        ArmResult(
            arm_id="replay_q",
            training_return=0.0,
            evaluation_matrix=_matrix(rows),
            continual_evaluation=0.0,
            isolated_forgetting=0.0,
            isolated_forward_transfer=0.0,
            receipt=_receipt(),
            candidate_eligible=True,
        )
    assert time.perf_counter() - started < 0.5


def test_metrics_reject_oversized_matrix_before_asarray() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="checkpoint rows"):
        _metrics(_matrix(8_000_000))
    assert time.perf_counter() - started < 0.5
