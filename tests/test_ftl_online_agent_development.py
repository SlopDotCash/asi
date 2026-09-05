from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks.ftl_online_agent_development import (
    ACTION_DELTAS,
    ARM_IDS,
    FROZEN_GOALS,
    FROZEN_SEEDS,
    OFFICIAL_CODE_REVISION,
    _mpc_action,
    run_development_lane,
    validate_result,
)

pytestmark = pytest.mark.integration

# Every frozen task starts at the origin, each goal sits two unit moves away,
# and no action holds position.  The best reachable cost is therefore one per
# odd step and zero per even step: exactly two per four-step task.
_OPTIMAL_DEFAULT_RETURN = -2.0 * len(FROZEN_GOALS)


def test_end_to_end_lane_has_matched_axes_controls_and_exact_receipts() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[0], steps_per_task=2, planning_horizon=1)
    assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
    assert result.development_only and not result.scientific_promotion_allowed
    assert result.negative_results_must_be_retained and not result.historical_ftl_claim_reused
    assert result.allowed_boundary_information == ()
    assert result.allowed_task_information == ("current_goal",)
    assert OFFICIAL_CODE_REVISION.endswith("a4fdb3b94a07a40d76e28d3aeab0f8ca97519dad")
    for arm in result.arms:
        assert arm.receipt.environment_steps == 6
        assert arm.receipt.planner_candidates == 24
    assert result.arms[0].receipt.model_queries == result.arms[1].receipt.model_queries == 30
    assert result.arms[2].receipt.model_queries == 24
    assert result.arms[0].receipt.model_updates == 6
    assert result.arms[1].receipt.model_updates == result.arms[2].receipt.model_updates == 0
    assert result.arms[0].receipt.logical_compute_units == 66
    assert result.arms[1].receipt.logical_compute_units == 60
    assert result.arms[2].receipt.logical_compute_units == 54
    assert result.arms[2].candidate_eligible is False


def test_sparse_model_predict_and_update_are_jit_eager_numerically_equivalent() -> None:
    # The end-to-end lane invokes the model's jitted methods; disabling JIT checks its eager path.
    compiled = run_development_lane(seed=FROZEN_SEEDS[1], steps_per_task=1, planning_horizon=1)
    with jax.disable_jit():
        eager = run_development_lane(seed=FROZEN_SEEDS[1], steps_per_task=1, planning_horizon=1)
    for compiled_arm, eager_arm in zip(compiled.arms, eager.arms, strict=True):
        assert compiled_arm.task_returns == eager_arm.task_returns
        assert compiled_arm.prequential_squared_errors == pytest.approx(
            eager_arm.prequential_squared_errors,
            rel=1e-7,
            abs=1e-12,
        )


@pytest.mark.parametrize("seed", [True, -1, 0, FROZEN_SEEDS[-1] + 1])
def test_runner_rejects_hostile_or_unfrozen_seeds(seed: object) -> None:
    with pytest.raises(ValueError):
        run_development_lane(seed=seed, steps_per_task=1, planning_horizon=1)


def test_validator_rejects_promotion_counter_forgery_and_extra_fields() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[0], steps_per_task=1, planning_horizon=1)
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_result(dataclasses.replace(result, scientific_promotion_allowed=True))
    receipt = dataclasses.replace(result.arms[0].receipt, model_queries=1)
    forged = dataclasses.replace(
        result,
        arms=(dataclasses.replace(result.arms[0], receipt=receipt), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="planner receipt"):
        validate_result(forged)
    payload = dataclasses.asdict(result)
    payload["unexpected"] = 1
    with pytest.raises(ValueError, match="exact mapping"):
        validate_result(payload)

    bad_bytes = dataclasses.replace(result.arms[0].receipt, persistent_bytes=1)
    forged_bytes = dataclasses.replace(
        result,
        arms=(dataclasses.replace(result.arms[0], receipt=bad_bytes), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="persistent-byte"):
        validate_result(forged_bytes)

    missing_metric = dataclasses.replace(
        result.arms[0], prequential_squared_errors=result.arms[0].prequential_squared_errors[:-1]
    )
    forged_metrics = dataclasses.replace(result, arms=(missing_metric, *result.arms[1:]))
    with pytest.raises(ValueError, match="metric count"):
        validate_result(forged_metrics)

    forged_identity = dataclasses.replace(result.identity, paper_registry_sha256="0" * 64)
    with pytest.raises(ValueError, match="current source/runtime/registries"):
        validate_result(dataclasses.replace(result, identity=forged_identity))


def test_mechanism_off_is_causal_and_does_not_adopt_updates() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[2], steps_per_task=3, planning_horizon=1)
    enabled, disabled, _ = result.arms
    assert enabled.receipt.persistent_bytes == disabled.receipt.persistent_bytes
    assert enabled.prequential_squared_errors != disabled.prequential_squared_errors
    assert enabled.task_returns != disabled.task_returns


def test_workload_action_registry_is_read_only() -> None:
    with pytest.raises(ValueError):
        ACTION_DELTAS[0, 0] = 7


def _grid_predict(observation: np.ndarray, action: int) -> np.ndarray:
    return np.asarray(observation + ACTION_DELTAS[action], dtype=np.float32)


def test_planner_maximizes_the_summed_horizon_return_not_the_terminal_state() -> None:
    """arXiv:2507.09177v1 Eq. 2 sums the H-step reward over the imagined rollout.

    This deterministic model separates the two objectives.  Action ``0`` is a
    detour that is far from the goal after one step and exactly on it after two;
    every other action parks one unit away.  Terminal-state scoring prefers the
    detour, the summed horizon return prefers parking.
    """

    def detour_predict(observation: np.ndarray, action: int) -> np.ndarray:
        if action == 0:
            far = float(observation[0]) == 0.0
            return np.asarray((10.0 if far else 0.0, 0.0), dtype=np.float32)
        return np.asarray((1.0, 0.0), dtype=np.float32)

    observation = np.zeros(2, dtype=np.float32)
    goal = np.zeros(2, dtype=np.float32)
    action, queries, candidates = _mpc_action(observation, goal, 2, detour_predict)
    assert (queries, candidates) == (32, 16)
    assert action == 1


def test_planner_horizon_one_reduces_to_the_single_imagined_step() -> None:
    """One imagined step makes the summed and terminal objectives identical."""
    observation = np.zeros(2, dtype=np.float32)
    for goal_tuple in FROZEN_GOALS:
        goal = np.asarray(goal_tuple, dtype=np.float32)
        action, queries, candidates = _mpc_action(observation, goal, 1, _grid_predict)
        terminal_scores = [
            -float(np.sum((_grid_predict(observation, candidate) - goal) ** 2))
            for candidate in range(4)
        ]
        assert (queries, candidates) == (4, 4)
        assert action == int(np.argmax(terminal_scores))


def test_privileged_control_bounds_every_arm_at_the_frozen_default_horizon() -> None:
    """The lane's defaults are its frozen protocol; run them, not a shrunk proxy."""
    for seed in FROZEN_SEEDS:
        result = run_development_lane(seed=seed)
        returns = {arm.arm_id: sum(arm.task_returns) for arm in result.arms}
        assert result.planning_horizon == 2
        assert all(
            arm.receipt.environment_steps == 4 * len(FROZEN_GOALS) for arm in result.arms
        )
        privileged = returns["privileged_dynamics_mpc"]
        assert privileged == pytest.approx(_OPTIMAL_DEFAULT_RETURN)
        assert privileged >= returns["sparse_ftl_online"]
        assert privileged >= returns["sparse_ftl_frozen"]


def test_deeper_planning_never_degrades_the_privileged_dynamics_control() -> None:
    """Exact dynamics plus more lookahead cannot lower the control's own return."""
    privileged_returns = []
    for horizon in (1, 2, 3):
        result = run_development_lane(seed=FROZEN_SEEDS[0], planning_horizon=horizon)
        arm = next(x for x in result.arms if x.arm_id == "privileged_dynamics_mpc")
        privileged_returns.append(sum(arm.task_returns))
    assert privileged_returns == pytest.approx([_OPTIMAL_DEFAULT_RETURN] * 3)
