from __future__ import annotations

import numpy as np
import pytest

import alberta_framework.evaluation.replay_frozen_ipmnist_campaign as campaign
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig


def _tiny_plan() -> campaign.ReplayFrozenCampaignPlan:
    return campaign.ReplayFrozenCampaignPlan(
        seeds=(1_573_901,),
        config=IPMNISTConfig(
            n_tasks=1, task_length=4, input_dim=4, hidden1=3, hidden2=2, n_classes=2
        ),
    )


def _data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [-1.0, -0.5, 0.5, 1.0],
                [1.0, 0.5, -0.5, -1.0],
                [-0.5, 1.0, -1.0, 0.5],
                [0.5, -1.0, 1.0, -0.5],
            ],
            dtype=np.float32,
        ),
        np.asarray([0, 1, 0, 1], dtype=np.int32),
    )


def test_frozen_plan_binds_fresh_seeds_all_arms_and_resource_allowances() -> None:
    payload = campaign.campaign_plan_payload(campaign.FROZEN_REPLAY_FROZEN_PLAN)
    assert payload["execution_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert payload["negative_outcomes_retained"] is True
    assert payload["seeds"] == [1_573_001, 1_573_002, 1_573_003]
    assert payload["arms"] == list(campaign.REPLAY_FROZEN_ARMS)
    assert set(payload["resource_allowances"]) == set(campaign.REPLAY_FROZEN_ARMS)
    assert payload["matched_axes"] == [
        "seed",
        "updates",
        "observations",
        "example_order",
        "task_permutations",
    ]


def test_execution_fails_before_data_or_arm_work_without_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("runner must not execute")

    monkeypatch.setattr(campaign, "run_screening_config", forbidden)
    with pytest.raises(PermissionError, match="authorization"):
        campaign.execute_replay_frozen_seed(object(), object(), seed=1_573_001)
    assert calls == 0


def test_authorized_tiny_seed_executes_and_validates_all_eight_matched_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    monkeypatch.setattr(campaign, "FROZEN_REPLAY_FROZEN_PLAN", plan)
    monkeypatch.setattr(campaign, "EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(campaign, "AUTHORIZATION_TRANSITION_APPROVED", True)
    x, y = _data()

    receipts = campaign.execute_replay_frozen_seed(x, y, seed=plan.seeds[0])

    assert tuple(receipt["arm"] for receipt in receipts) == campaign.REPLAY_FROZEN_ARMS
    assert all(receipt["seed"] == plan.seeds[0] for receipt in receipts)
    assert all(receipt["outcome"] == "inconclusive" for receipt in receipts)
    assert all(receipt["scientific_promotion_allowed"] is False for receipt in receipts)
