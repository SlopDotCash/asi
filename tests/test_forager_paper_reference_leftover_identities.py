"""Leftover-identity gates for forager paper-reference records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.benchmarks.forager import (
    PaperBaseline,
    PaperReferenceTarget,
    paper_reference_targets,
)


def _legal_target() -> PaperReferenceTarget:
    return PaperReferenceTarget("PPO", "mean_ewm_reward", 1.3)


def _legal_baseline() -> PaperBaseline:
    return PaperBaseline(
        name="PPO",
        family="ppo",
        role="sota",
        state_construction="pixels",
        selected_hyperparameters={"step_size": 0.1},
        in_tree_implementation=True,
        source="https://arxiv.org/abs/2605.01131",
    )


def test_paper_reference_target_rejects_leftover_identities() -> None:
    """Public paper-reference records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="central_estimate"):
        PaperReferenceTarget("PPO", "mean_ewm_reward", True)
    with pytest.raises(ValueError, match="central_estimate"):
        PaperReferenceTarget("PPO", "mean_ewm_reward", float("nan"))
    with pytest.raises(ValueError, match="privileged"):
        PaperReferenceTarget("PPO", "mean_ewm_reward", 1.3, privileged=1)

    legal = _legal_target()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"central_estimate": 1.3' in dumped
    assert '"privileged": false' in dumped
    assert '"central_estimate": true' not in dumped
    assert '"privileged": 1' not in dumped


def test_paper_baseline_rejects_leftover_identities() -> None:
    """Published baseline contracts must not keep leftover bool identities."""

    with pytest.raises(ValueError, match="in_tree_implementation"):
        PaperBaseline(
            name="PPO",
            family="ppo",
            role="sota",
            state_construction="pixels",
            selected_hyperparameters={"step_size": 0.1},
            in_tree_implementation=1,
            source="https://arxiv.org/abs/2605.01131",
        )

    legal = _legal_baseline()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"in_tree_implementation": true' in dumped
    assert '"in_tree_implementation": 1' not in dumped


def test_paper_reference_targets_remain_legal() -> None:
    targets = paper_reference_targets("relearning")
    rtu = next(item for item in targets if item.method == "RTU-PPO")
    assert rtu.central_estimate == pytest.approx(1.3)
    dumped = json.dumps(rtu.to_dict(), allow_nan=False)
    assert '"central_estimate": 1.3' in dumped
