"""Leftover-identity gates for forager paper-reference records."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alberta_framework.benchmarks.forager import (
    ForagerEnvConfig,
    ForagerRunResult,
    PaperBaseline,
    PaperForagerProtocol,
    PaperReferenceTarget,
    paper_protocol,
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


def _legal_run_result(**overrides: object) -> ForagerRunResult:
    payload: dict[str, object] = {
        "agent": "toy",
        "privileged": False,
        "seed": 0,
        "steps": 10,
        "total_reward": 1.0,
        "mean_reward": 0.1,
        "final_window_mean_reward": 0.1,
        "final_ewm_reward": 0.1,
        "mean_ewm_reward": 0.1,
        "fov_last_10pct_ema_auc": 0.1,
        "mean_biome_regret": 0.0,
        "final_biome_regret": 0.0,
        "curve_steps": (1, 10),
        "curve_ewm_reward": (0.1, 0.1),
        "curve_window_reward": (0.1, 0.1),
        "duration_s": 1.0,
        "frames_per_second": 10.0,
        "environment": {"env_id": "fake"},
        "metric_contract": {"metric": "mean_reward"},
        "agent_metadata": {"name": "toy"},
    }
    payload.update(overrides)
    return ForagerRunResult(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_seeds", True),
        ("evaluation_seeds", False),
        ("evaluation_seeds", 0),
        ("final_window_steps", True),
        ("final_window_steps", 0),
        ("tuning_seeds", True),
        ("evaluation_steps", True),
        ("single_stream", 1),
        ("confidence", True),
        ("ewm_decay", True),
    ],
)
def test_paper_forager_protocol_rejects_bool_counts_and_windows(
    field: str, value: object
) -> None:
    legal = paper_protocol()
    kwargs = {name: getattr(legal, name) for name in legal.__dataclass_fields__}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        PaperForagerProtocol(**kwargs)  # type: ignore[arg-type]


def test_paper_forager_protocol_keeps_integer_json_and_seed_range() -> None:
    protocol = paper_protocol("relearning")
    dumped = json.dumps(protocol.to_dict(), allow_nan=False)
    assert '"evaluation_seeds": 30' in dumped
    assert '"final_window_steps": 100000' in dumped
    assert '"evaluation_seeds": true' not in dumped
    assert '"final_window_steps": true' not in dumped
    seeds = range(
        protocol.evaluation_seed_start,
        protocol.evaluation_seed_start + protocol.evaluation_seeds,
    )
    assert len(tuple(seeds)) == 30


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", True),
        ("steps", False),
        ("steps", 0),
        ("seed", True),
        ("seed", -1),
        ("privileged", 1),
        ("mean_reward", True),
        ("duration_s", True),
        ("duration_s", float("inf")),
    ],
)
def test_forager_run_result_rejects_bool_seed_and_steps(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _legal_run_result(**{field: value})


def test_forager_run_result_keeps_integer_json() -> None:
    result = _legal_run_result()
    dumped = json.dumps(result.to_dict(), allow_nan=False)
    assert '"seed": 0' in dumped
    assert '"steps": 10' in dumped
    assert '"seed": true' not in dumped
    assert '"steps": true' not in dumped
    assert '"privileged": false' in dumped


def test_forager_run_result_preserves_zero_based_historical_curve_grid() -> None:
    result = _legal_run_result(
        curve_steps=(0, 10),
        curve_window_reward=(),
        environment={
            "runtime": "historical_numpy_forager",
            "pairable_with_current_foragax": False,
        },
        metric_contract={
            "stored_curve": "unadjusted_ema_then_subsample",
            "raw_reward_metrics_available": False,
        },
        agent_metadata={
            "result_source": "official_fov_sqlite",
            "raw_rewards_available": False,
        },
    )
    assert result.curve_steps == (0, 10)
    assert result.curve_window_reward == ()


def test_run_result_rejects_hostile_scalar_subclasses_before_hooks() -> None:
    calls = 0

    class HostileInt(int):
        def __int__(self) -> int:
            nonlocal calls
            calls += 1
            raise AssertionError("integer conversion hook reached")

    class HostileFloat(float):
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("float conversion hook reached")

    with pytest.raises(ValueError, match="seed"):
        _legal_run_result(seed=HostileInt(1))
    with pytest.raises(ValueError, match="mean_reward"):
        _legal_run_result(mean_reward=HostileFloat(1.0))
    assert calls == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"curve_steps": ()},
        {"curve_steps": (-1, 10)},
        {"curve_steps": (0, 10)},
        {"curve_steps": (1, 1)},
        {"curve_steps": (1, 11)},
        {"curve_ewm_reward": (0.1,)},
        {"curve_window_reward": (0.1,)},
        {"curve_ewm_reward": (), "curve_window_reward": ()},
        {"duration_s": -1.0},
        {"frames_per_second": -1.0},
    ],
)
def test_run_result_rejects_cross_field_and_domain_drift(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _legal_run_result(**overrides)


def test_protocol_rejects_hostile_string_subclasses_before_comparison() -> None:
    calls = 0

    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("string comparison hook reached")

        __hash__ = str.__hash__

    legal = paper_protocol()
    kwargs = {name: getattr(legal, name) for name in legal.__dataclass_fields__}
    for field in ("preset", "primary_metric"):
        hostile = dict(kwargs)
        hostile[field] = HostileString(str(kwargs[field]))
        with pytest.raises(ValueError, match=field):
            PaperForagerProtocol(**hostile)  # type: ignore[arg-type]
    assert calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("final_window_steps", 10_000_001),
        ("frozen_ablation_after_steps", 10_000_001),
        ("hidden_switch_interval_steps", 10_000_001),
        ("evaluation_seed_start", 2**31 - 20),
    ],
)
def test_protocol_rejects_cross_field_resource_drift(field: str, value: int) -> None:
    legal = paper_protocol()
    kwargs = {name: getattr(legal, name) for name in legal.__dataclass_fields__}
    kwargs[field] = value
    with pytest.raises(ValueError):
        PaperForagerProtocol(**kwargs)  # type: ignore[arg-type]


def test_environment_rejects_hostile_container_and_scalar_subclasses() -> None:
    with pytest.raises(ValueError, match="aperture_size"):
        ForagerEnvConfig(aperture_size=np.int32(9))  # type: ignore[arg-type]

    class MappingSubclass(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="actual dict"):
        ForagerEnvConfig(extra_kwargs=MappingSubclass())
