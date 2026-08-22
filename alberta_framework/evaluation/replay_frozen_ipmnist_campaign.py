"""Prospective, permanently nonpromoting replay/frozen IPMNIST campaign.

The plan is frozen here before execution.  Its authorization switches are
deliberately false: importing this module or invoking the executor cannot
consume a development seed until a maintainer explicitly approves both the
run and the source-changing authorization transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.benchmarks.ipmnist_screening import (
    replay_frozen_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    expected_resources_for_result,
    validate_matched_replay_frozen_results,
)

PLAN_SCHEMA: Final = "asi.replay-frozen-ipmnist.campaign-plan.v1"
PLAN_ID: Final = "issue-1573.replay-frozen-ceiling.development.v1"
EXECUTION_AUTHORIZED: Final = False
AUTHORIZATION_TRANSITION_APPROVED: Final = False
REPLAY_FROZEN_ARMS: Final = (
    "replay_context_mechanism_off",
    "replay_gradient_only",
    "replay_context_only",
    "replay_context_full",
    "randumb_random_features",
    "ranpac_random_projection",
    "prol_prompt_mechanism_off",
    "prol_prompt_proxy",
)
_FAMILIES: Final = MappingProxyType(
    {
        "replay_context_mechanism_off": "replay",
        "replay_gradient_only": "replay",
        "replay_context_only": "replay",
        "replay_context_full": "replay",
        "randumb_random_features": "randumb",
        "ranpac_random_projection": "ranpac",
        "prol_prompt_mechanism_off": "prol",
        "prol_prompt_proxy": "prol",
    }
)


@dataclass(frozen=True)
class ReplayFrozenCampaignPlan:
    """Exact matched development workload selected before execution."""

    seeds: tuple[int, ...]
    config: IPMNISTConfig

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= 32:
            raise ValueError("seeds must be a bounded exact tuple")
        seeds = tuple(require_jax_seed(seed, name="seed") for seed in self.seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if type(self.config) is not IPMNISTConfig:
            raise ValueError("config must be an exact IPMNISTConfig")
        config = IPMNISTConfig(**self.config.to_config())
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "config", config)


FROZEN_REPLAY_FROZEN_PLAN: Final = ReplayFrozenCampaignPlan(
    seeds=(1_573_001, 1_573_002, 1_573_003),
    config=IPMNISTConfig(n_tasks=10, task_length=128),
)


def _family_resources(plan: ReplayFrozenCampaignPlan, arm: str) -> dict[str, int]:
    config = plan.config
    return expected_resources_for_result(
        _FAMILIES[arm],
        config.n_steps,
        config.input_dim,
        config.hidden1,
        config.hidden2,
        config.n_classes,
    )


def campaign_plan_payload(plan: ReplayFrozenCampaignPlan) -> dict[str, object]:
    """Return the complete reviewable plan without executing any arm."""
    if type(plan) is not ReplayFrozenCampaignPlan:
        raise ValueError("plan must be an exact ReplayFrozenCampaignPlan")
    checked = ReplayFrozenCampaignPlan(seeds=plan.seeds, config=plan.config)
    config = checked.config
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": PLAN_ID,
        "issue": 1573,
        "execution_authorized": EXECUTION_AUTHORIZED,
        "authorization_transition_approved": AUTHORIZATION_TRANSITION_APPROVED,
        "seeds": list(checked.seeds),
        "arms": list(REPLAY_FROZEN_ARMS),
        "ordering": "seed-major-then-frozen-arm-order",
        "workload": config.to_config(),
        "matched_axes": [
            "seed",
            "updates",
            "observations",
            "example_order",
            "task_permutations",
        ],
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "expected_shards": len(checked.seeds) * len(REPLAY_FROZEN_ARMS),
        "expected_observations_per_arm": config.n_steps,
        "expected_updates_per_arm": config.n_steps,
        "resource_allowances": {
            arm: _family_resources(checked, arm) for arm in REPLAY_FROZEN_ARMS
        },
        "timing_is_telemetry_only": True,
        "output_namespace": "outputs/ipmnist_replay_frozen/development.v1",
        "append_only_publication_required": True,
        "negative_outcomes_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_rule": "descriptive-only; no promotion threshold",
    }


def execute_replay_frozen_seed(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    *,
    seed: int,
) -> tuple[dict[str, object], ...]:
    """Execute one exact eight-arm shard group after explicit authorization."""
    if EXECUTION_AUTHORIZED is not True or AUTHORIZATION_TRANSITION_APPROVED is not True:
        raise PermissionError("issue #1573 campaign execution requires explicit authorization")
    plan = FROZEN_REPLAY_FROZEN_PLAN
    resolved_seed = require_jax_seed(seed, name="seed")
    if resolved_seed not in plan.seeds:
        raise ValueError("seed is outside the frozen issue #1573 roster")
    receipts = tuple(
        replay_frozen_development_result_payload(
            run_screening_config(
                data_x,
                data_y,
                screening_spec(arm),
                resolved_seed,
                plan.config,
            ),
            outcome="inconclusive",
        )
        for arm in REPLAY_FROZEN_ARMS
    )
    return validate_matched_replay_frozen_results(receipts)
