"""Cheap contracts for the permanently nonpromoting reference-life scorecard."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks import reference_life_scorecard as scorecard
from alberta_framework.benchmarks.reference_life_scorecard import (
    ARM_ROSTER,
    ENVIRONMENT_ROSTER,
    SEED_ROSTER,
    ReferenceLifeDevelopmentPlan,
    StreamingRunSummary,
    build_development_plan,
    canonical_json_bytes,
    estimate_jax_resources,
    parameter_change_check,
    write_new_json,
)
from alberta_framework.reference_agent import ArrayValue
from alberta_framework.reference_life_controls import (
    DifferentialSARSAReferenceConfig,
    DiscountedSARSAReferenceConfig,
)
from alberta_framework.streams.closed_loop import SwitchingTwoStateConfig

pytestmark = pytest.mark.unit


def test_scorecard_plan_import_does_not_require_linux_checkpoint_runtime() -> None:
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "fcntl":
        raise ModuleNotFoundError("blocked nonportable fcntl")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from alberta_framework.benchmarks.reference_life_scorecard import (
    build_development_plan,
    iter_run_specs,
)
assert len(iter_run_specs(build_development_plan())) == 144
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_publication_fails_closed_without_posix_fcntl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import alberta_framework.reference_life_checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "fcntl", None)
    with pytest.raises(OSError, match="requires POSIX fcntl support"):
        checkpoint_module.save_reference_life_checkpoint(  # type: ignore[arg-type]
            object(), object(), tmp_path
        )


def test_fixed_plan_is_immutable_canonical_and_explicit() -> None:
    plan = build_development_plan()
    payload = plan.to_payload()

    assert dataclasses.is_dataclass(plan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.plan_sha256 = "0" * 64  # type: ignore[misc]
    assert plan.seeds == tuple(range(70_000, 70_012)) == SEED_ROSTER
    assert plan.arms == ARM_ROSTER
    assert plan.environments == ENVIRONMENT_ROSTER
    assert payload["evidence_policy"]["permanently_nonpromoting"] is True
    assert payload["evidence_policy"]["scientific_promotion_allowed"] is False
    assert payload["protocols"]["switching_two_state"]["horizon"] == 4_000
    assert payload["protocols"]["switching_two_state"]["phase_length"] == 250
    assert payload["protocols"]["switching_two_state"]["post_switch_window"] == 50
    assert payload["protocols"]["riverswim"]["horizon"] == 20_000
    assert payload["protocols"]["riverswim"]["n_states"] == 6
    assert payload["protocols"]["riverswim"]["early_window"] == 2_000
    assert payload["protocols"]["riverswim"]["late_window"] == 2_000
    assert canonical_json_bytes(payload) == canonical_json_bytes(
        ReferenceLifeDevelopmentPlan.from_payload(payload).to_payload()
    )
    assert payload["plan_sha256"] == scorecard.REFERENCE_LIFE_SCORECARD_PLAN_V1_SHA256

    payload["seed_roster"][0] = 1
    assert plan.seeds == SEED_ROSTER, "returned JSON must not alias immutable plan state"


def test_plan_rejects_tampering_even_if_digest_is_left_unchanged() -> None:
    payload = build_development_plan().to_payload()
    payload["protocols"]["riverswim"]["horizon"] -= 1
    with pytest.raises(ValueError, match="canonical fixed development plan"):
        ReferenceLifeDevelopmentPlan.from_payload(payload)


def test_plan_rejects_boolean_integer_type_confusion() -> None:
    payload = build_development_plan().to_payload()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="canonical fixed development plan"):
        ReferenceLifeDevelopmentPlan.from_payload(payload)


def test_cyclic_order_is_explicit_and_balanced() -> None:
    plan = build_development_plan()
    observed = [plan.arm_order(seed) for seed in SEED_ROSTER]
    assert observed[0] == ARM_ROSTER
    assert observed[1] == ARM_ROSTER[1:] + ARM_ROSTER[:1]
    assert observed[6] == ARM_ROSTER
    for position in range(len(ARM_ROSTER)):
        assert sorted(order[position] for order in observed) == sorted(list(ARM_ROSTER) * 2)


def test_streaming_switching_summary_keeps_only_fixed_aba_windows() -> None:
    summary = StreamingRunSummary.for_switching(
        horizon=8,
        phase_length=2,
        post_switch_window=2,
    )
    for index in range(8):
        summary.observe(
            reward=float(index),
            oracle_reward=10.0,
            regime_id=(index // 2) % 2,
            parameters_changed=index == 3,
            next_state_index=index % 2,
        )

    result = summary.finalize()
    assert result["accepted_events"] == 8
    assert result["parameter_change_events"] == 1
    assert result["windows"] == {
        "initial_a": {
            "event_count": 2,
            "reward_sum": 1.0,
            "mean_reward": 0.5,
            "mean_oracle_regret": 9.5,
        },
        "first_b": {
            "event_count": 2,
            "reward_sum": 5.0,
            "mean_reward": 2.5,
            "mean_oracle_regret": 7.5,
        },
        "return_a": {
            "event_count": 2,
            "reward_sum": 9.0,
            "mean_reward": 4.5,
            "mean_oracle_regret": 5.5,
        },
    }
    assert not hasattr(summary, "events")
    with pytest.raises(ValueError, match="horizon"):
        summary.observe(
            reward=0.0,
            oracle_reward=1.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )


def test_streaming_river_summary_tracks_early_late_and_high_end_visits() -> None:
    summary = StreamingRunSummary.for_riverswim(
        horizon=6,
        early_window=2,
        late_window=2,
        n_states=3,
    )
    for index, state_index in enumerate((0, 1, 2, 2, 1, 2)):
        summary.observe(
            reward=float(index + 1),
            oracle_reward=7.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=state_index,
        )
    result = summary.finalize()
    assert result["windows"]["early"]["reward_sum"] == 3.0
    assert result["windows"]["late"]["reward_sum"] == 11.0
    assert result["high_end_visit_count"] == 3
    assert result["high_end_visit_rate"] == 0.5


@dataclasses.dataclass(frozen=True)
class _TinyState:
    weights: Any
    counter: Any


def test_resource_estimate_is_explicitly_a_pytree_estimate() -> None:
    estimate = estimate_jax_resources(
        _TinyState(
            weights=jnp.zeros((2, 3), dtype=jnp.float32),
            counter=jnp.asarray(0, dtype=jnp.int32),
        )
    )
    assert estimate["persistent_jax_array_bytes"] == 28
    assert estimate["persistent_jax_array_scalar_count"] == 7
    assert estimate["trainable_scalar_count_estimate"] == 6
    assert estimate["trainable_scalar_count_method"] == (
        "floating_jax_pytree_leaves_upper_bound"
    )


def test_resource_estimate_counts_cached_array_value_payloads() -> None:
    cached = ArrayValue(
        semantic_id="test.cached_observation.v1",
        dtype="float32",
        shape=(2,),
        payload=np.asarray([1.0, 0.0], dtype="<f4").tobytes(),
    )
    estimate = estimate_jax_resources(
        {
            "state": _TinyState(
                weights=jnp.zeros((2, 3), dtype=jnp.float32),
                counter=jnp.asarray(0, dtype=jnp.int32),
            ),
            "cached": cached,
        }
    )
    assert estimate["persistent_jax_array_leaves"] == 3
    assert estimate["persistent_jax_array_scalar_count"] == 9
    assert estimate["persistent_jax_array_bytes"] == 36
    assert estimate["trainable_scalar_count_estimate"] == 8


def test_discounted_sarsa_persistent_resources_are_fixed_from_initialization() -> None:
    plan = build_development_plan()
    spec = next(
        item
        for item in scorecard.iter_run_specs(plan)
        if item.environment_kind == "switching_two_state"
        and item.arm == "sarsa"
        and item.seed == SEED_ROSTER[0]
    )
    runner = scorecard.build_scorecard_runner(plan, spec)
    initial = runner.init()
    initial_resources = scorecard._agent_resource_payload(
        runner.agent_adapter,
        initial.agent_state,
    )

    step = runner.step(initial)

    assert step.accepted
    assert scorecard._agent_resource_payload(
        runner.agent_adapter,
        step.state.agent_state,
    ) == initial_resources


@pytest.mark.parametrize(
    ("arm", "changes", "passed"),
    [
        ("prototype", 2, True),
        ("prototype", 0, False),
        ("prototype_frozen", 0, True),
        ("prototype_frozen", 1, False),
        ("random", 0, True),
        ("privileged_oracle", 0, True),
    ],
)
def test_parameter_change_checks_are_fail_closed(
    arm: str, changes: int, passed: bool
) -> None:
    assert parameter_change_check(arm, changes)["passed"] is passed


def _summary_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offsets = {
        "random": 0.0,
        "privileged_oracle": 100.0,
        "prototype": 50.0,
        "prototype_frozen": 0.0,
        "differential_sarsa": 40.0,
        "sarsa": 30.0,
    }
    for environment in ENVIRONMENT_ROSTER:
        for arm in ARM_ROSTER:
            for seed in SEED_ROSTER:
                records.append(
                    {
                        "environment_kind": environment,
                        "arm": arm,
                        "seed": seed,
                        "status": "completed",
                        "outcome": {"reward_sum": offsets[arm]},
                    }
                )
    return records


def test_summary_normalizes_within_environment_and_never_pools() -> None:
    records = _summary_records()
    summary = scorecard._summarize_validated_run_records(
        build_development_plan(), list(reversed(records))
    )
    assert summary["status"] == "development_scorecard_complete"
    assert summary["cross_environment_pooled_score"] is None
    assert summary["cross_environment_pooling_forbidden"] is True
    future_gate = summary["pareto_resource_decision"][
        "future_gate_if_latency_is_qualified_under_a_new_protocol"
    ]
    assert future_gate[
        "requires_no_qualifying_sarsa_utility_noninferior_on_both_environments"
    ] is True
    for environment in ENVIRONMENT_ROSTER:
        environment_summary = summary["environments"][environment]
        assert environment_summary["normalization"]["scale"] == 100.0
        assert environment_summary["arms"]["random"]["normalized_score_mean"] == 0.0
        assert environment_summary["arms"]["privileged_oracle"][
            "normalized_score_mean"
        ] == 1.0
        assert environment_summary["arms"]["differential_sarsa"][
            "paired_t_lcb_95"
        ] > 0.10
    assert canonical_json_bytes(summary) == canonical_json_bytes(
        scorecard._summarize_validated_run_records(build_development_plan(), records)
    )


def test_public_summary_rejects_unvalidated_skeletal_records() -> None:
    with pytest.raises(ValueError, match="fields|record|contract"):
        scorecard.summarize_run_records(build_development_plan(), _summary_records())


def test_summary_retains_failures_and_reports_valid_baseline_failure() -> None:
    records = _summary_records()
    for record in records:
        if record["arm"] in ("differential_sarsa", "sarsa"):
            record["outcome"]["reward_sum"] = 5.0
    records[0] = {
        **records[0],
        "status": "failed",
        "outcome": None,
        "failure": {"stage": "step", "type": "RuntimeError", "message": "boom"},
    }
    summary = scorecard._summarize_validated_run_records(
        build_development_plan(), records
    )
    assert summary["failure_count"] == 1
    assert summary["failures"][0]["failure"]["message"] == "boom"
    assert summary["status"] == "valid_execution_failure"

    complete = [record for record in records if record["status"] == "completed"]
    complete.append(
        {
            **records[0],
            "status": "completed",
            "outcome": {"reward_sum": 0.0},
        }
    )
    summary = scorecard._summarize_validated_run_records(
        build_development_plan(), complete
    )
    assert summary["status"] == "valid_baseline_failure"


@pytest.mark.skipif(
    not hasattr(os, "O_TMPFILE"),
    reason="write_new_json publishes through Linux O_TMPFILE and linkat(AT_EMPTY_PATH)",
)
def test_new_json_publication_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "plan.json"
    payload = build_development_plan().to_payload()
    write_new_json(destination, payload)
    assert destination.read_bytes() == json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_new_json(destination, {"different": math.pi})


@pytest.mark.skipif(
    hasattr(os, "O_TMPFILE"),
    reason="Linux publishes the document instead of refusing",
)
def test_new_json_publication_fails_closed_without_o_tmpfile(tmp_path: Path) -> None:
    destination = tmp_path / "plan.json"
    with pytest.raises(OSError, match="requires Linux O_TMPFILE support"):
        write_new_json(destination, build_development_plan().to_payload())
    assert not destination.exists()


def test_new_json_publication_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OSError):
        write_new_json(linked_parent / "plan.json", build_development_plan().to_payload())
    assert not (real_parent / "plan.json").exists()


def test_failed_shard_is_retained_and_digest_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"schema": "test.identity.v1", "value": "fixed"}
    monkeypatch.setattr(scorecard, "_checkpoint_source_identity", lambda: identity)
    monkeypatch.setattr(scorecard, "_checkpoint_runtime_identity", lambda: identity)
    monkeypatch.setattr(scorecard, "_checkpoint_dependency_identity", lambda: identity)

    def fail_build(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("intentional cheap build failure")

    monkeypatch.setattr(scorecard, "build_scorecard_runner", fail_build)
    plan = build_development_plan()
    spec = scorecard.iter_run_specs(plan)[0]
    record = scorecard.run_scorecard_shard(plan, spec)
    assert record["status"] == "failed"
    assert record["failure"] == {
        "stage": "build",
        "type": "RuntimeError",
        "message": "intentional cheap build failure",
        "accepted_events": 0,
    }
    assert record["partial_outcome"]["summary_mode"] == (
        "streaming_o1_no_retained_events"
    )
    assert scorecard.validate_scorecard_run_record(record)["valid"] is True

    tampered = json.loads(json.dumps(record))
    tampered["failure"]["message"] = "forged"
    with pytest.raises(ValueError, match="content digest mismatch"):
        scorecard.validate_scorecard_run_record(tampered)


def _completed_outcome(
    plan: ReferenceLifeDevelopmentPlan,
    spec: scorecard.ScorecardRunSpec,
    runner: Any,
) -> dict[str, Any]:
    protocol = plan.protocol(spec.environment_kind)
    horizon = protocol["horizon"]
    if spec.environment_kind == "switching_two_state":
        oracle_reward = 1.0
        phase_counts = [horizon // 2, horizon // 2]
        windows = {
            name: {
                "event_count": protocol["post_switch_window"],
                "reward_sum": 0.0,
                "mean_reward": 0.0,
                "mean_oracle_regret": oracle_reward,
            }
            for name in ("initial_a", "first_b", "return_a")
        }
        high_end_visit_count = None
        high_end_visit_rate = None
    else:
        oracle_reward = runner.environment_adapter.manifest.config[
            "oracle_average_reward"
        ]
        phase_counts = [horizon, 0]
        windows = {
            name: {
                "event_count": protocol[f"{name}_window"],
                "reward_sum": 0.0,
                "mean_reward": 0.0,
                "mean_oracle_regret": oracle_reward,
            }
            for name in ("early", "late")
        }
        high_end_visit_count = 0
        high_end_visit_rate = 0.0
    changes = 1 if spec.arm in ("prototype", "differential_sarsa", "sarsa") else 0
    resource = scorecard._canonical_initial_resource_payload(
        spec.environment_kind,
        spec.arm,
    )
    oracle_reward_sum = oracle_reward * horizon
    return {
        "summary_mode": "streaming_o1_no_retained_events",
        "configured_horizon": horizon,
        "accepted_events": horizon,
        "reward_sum": 0.0,
        "mean_reward": 0.0,
        "oracle_reward_sum": oracle_reward_sum,
        "regret_sum": oracle_reward_sum,
        "parameter_change_events": changes,
        "phase_event_counts": phase_counts,
        "phase_reward_sums": [0.0, 0.0],
        "windows": windows,
        "high_end_visit_count": high_end_visit_count,
        "high_end_visit_rate": high_end_visit_rate,
        "environment_rng_cursor": horizon,
        "transcript_sha256": "1" * 64,
        "resources": {"initial": dict(resource), "final": dict(resource)},
        "parameter_change_check": parameter_change_check(spec.arm, changes),
    }


def _completed_record(
    plan: ReferenceLifeDevelopmentPlan,
    spec: scorecard.ScorecardRunSpec,
    *,
    runner: Any | None = None,
    identities: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    effective_runner = (
        scorecard.build_scorecard_runner(plan, spec) if runner is None else runner
    )
    source, runtime, dependencies = (
        scorecard._current_consistency_identities() if identities is None else identities
    )
    horizon = plan.protocol(spec.environment_kind)["horizon"]
    record = {
        "schema": scorecard.REFERENCE_LIFE_SCORECARD_RUN_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "schedule_index": spec.schedule_index,
        "environment_kind": spec.environment_kind,
        "arm": spec.arm,
        "seed": spec.seed,
        "lifecycle_id": spec.lifecycle_id,
        "evidence_policy": dict(scorecard.NONPROMOTING_POLICY),
        "source_identity": source,
        "runtime_identity": runtime,
        "dependency_identity": dependencies,
        "status": "completed",
        "failure": None,
        "resolved": scorecard._resolved_components(plan, spec, effective_runner),
        "outcome": _completed_outcome(plan, spec, effective_runner),
        "partial_outcome": None,
        "telemetry": {
            "policy": dict(scorecard.TELEMETRY_POLICY),
            "setup_seconds": 0.0,
            "cold_step_seconds": 0.0,
            "warmed_step_seconds_total": 0.0,
            "warmed_step_count": horizon - 1,
            "warmed_step_seconds_mean": 0.0,
            "total_seconds": 0.0,
        },
    }
    return scorecard._record_with_digest(record)


def _redigest(record: dict[str, Any]) -> None:
    record["record_sha256"] = scorecard._digest_excluding(record, "record_sha256")


@pytest.mark.parametrize("environment", ENVIRONMENT_ROSTER)
@pytest.mark.parametrize("arm", ARM_ROSTER)
def test_valid_completed_shards_cover_every_scheduled_component(
    environment: str,
    arm: str,
) -> None:
    plan = build_development_plan()
    spec = next(
        item
        for item in scorecard.iter_run_specs(plan)
        if item.environment_kind == environment
        and item.arm == arm
        and item.seed == SEED_ROSTER[0]
    )
    record = _completed_record(plan, spec)

    result = scorecard.validate_scorecard_run_record(record)

    assert result["valid"] is True
    assert result["status"] == "completed"
    assert result["permanently_nonpromoting"] is True


def test_sarsa_controls_are_constructed_from_plan_arm_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_development_plan()
    observed: dict[str, dict[str, Any]] = {}
    original_differential = DifferentialSARSAReferenceConfig.for_switching.__func__
    original_discounted = DiscountedSARSAReferenceConfig.for_switching.__func__

    def differential_spy(
        cls: type[DifferentialSARSAReferenceConfig],
        environment_config: SwitchingTwoStateConfig,
        **overrides: Any,
    ) -> DifferentialSARSAReferenceConfig:
        observed["differential_sarsa"] = dict(overrides)
        return original_differential(cls, environment_config, **overrides)

    def discounted_spy(
        cls: type[DiscountedSARSAReferenceConfig],
        environment_config: SwitchingTwoStateConfig,
        **overrides: Any,
    ) -> DiscountedSARSAReferenceConfig:
        observed["sarsa"] = dict(overrides)
        return original_discounted(cls, environment_config, **overrides)

    monkeypatch.setattr(
        DifferentialSARSAReferenceConfig,
        "for_switching",
        classmethod(differential_spy),
    )
    monkeypatch.setattr(
        DiscountedSARSAReferenceConfig,
        "for_switching",
        classmethod(discounted_spy),
    )
    for arm in ("differential_sarsa", "sarsa"):
        spec = next(
            item
            for item in scorecard.iter_run_specs(plan)
            if item.environment_kind == "switching_two_state" and item.arm == arm
        )
        scorecard.build_scorecard_runner(plan, spec)

    assert observed["differential_sarsa"] == plan.arm_definition(
        "differential_sarsa"
    )["config"]
    assert observed["sarsa"] == plan.arm_definition("sarsa")["config"]


def test_completed_shard_rejects_another_scheduled_arm_with_rehashed_bindings() -> None:
    plan = build_development_plan()
    random_spec = next(
        item
        for item in scorecard.iter_run_specs(plan)
        if item.environment_kind == "switching_two_state"
        and item.arm == "random"
        and item.seed == SEED_ROSTER[0]
    )
    oracle_spec = next(
        item
        for item in scorecard.iter_run_specs(plan)
        if item.environment_kind == "switching_two_state"
        and item.arm == "privileged_oracle"
        and item.seed == SEED_ROSTER[0]
    )
    random_record = _completed_record(plan, random_spec)
    oracle_record = _completed_record(plan, oracle_spec)
    swapped = copy.deepcopy(random_record["resolved"])
    swapped["arm_definition"] = plan.arm_definition("privileged_oracle")
    swapped["life_config"]["lifecycle_id"] = oracle_spec.lifecycle_id
    swapped["life_config_sha256"] = scorecard._sha256_json(swapped["life_config"])
    oracle_record["resolved"] = swapped
    _redigest(oracle_record)

    with pytest.raises(ValueError, match="scheduled arm|canonical resolved components"):
        scorecard.validate_scorecard_run_record(oracle_record)


def test_completed_shard_rejects_boolean_schedule_index_type_confusion() -> None:
    plan = build_development_plan()
    spec = scorecard.iter_run_specs(plan)[1]
    record = _completed_record(plan, spec)
    record["schedule_index"] = True
    _redigest(record)
    with pytest.raises(ValueError, match="canonical cyclic schedule"):
        scorecard.validate_scorecard_run_record(record)


def test_completed_shard_rejects_impossible_reward_lattice_total() -> None:
    plan = build_development_plan()
    spec = next(
        item
        for item in scorecard.iter_run_specs(plan)
        if item.environment_kind == "riverswim"
        and item.arm == "random"
        and item.seed == SEED_ROSTER[0]
    )
    record = _completed_record(plan, spec)
    outcome = record["outcome"]
    outcome["reward_sum"] = 0.001
    outcome["mean_reward"] = 0.001 / outcome["accepted_events"]
    outcome["phase_reward_sums"] = [0.001, 0.0]
    outcome["regret_sum"] = outcome["oracle_reward_sum"] - 0.001
    _redigest(record)
    with pytest.raises(ValueError, match="reward lattice"):
        scorecard.validate_scorecard_run_record(record)


@pytest.mark.parametrize("environment", ENVIRONMENT_ROSTER)
def test_failed_shard_rejects_window_reward_exceeding_partial_total(
    environment: str,
) -> None:
    plan = build_development_plan()
    spec = next(
        item
        for item in scorecard.iter_run_specs(plan)
        if item.environment_kind == environment
        and item.arm == "random"
        and item.seed == SEED_ROSTER[0]
    )
    record = _completed_record(plan, spec)
    horizon = plan.protocol(environment)["horizon"]
    oracle_reward = record["outcome"]["oracle_reward_sum"] / horizon
    window = {
        "event_count": 1,
        "reward_sum": 1.0,
        "mean_reward": 1.0,
        "mean_oracle_regret": oracle_reward - 1.0,
    }
    if environment == "switching_two_state":
        windows = {"initial_a": window, "first_b": None, "return_a": None}
        high_end_visit_count = None
        high_end_visit_rate = None
    else:
        windows = {"early": window, "late": None}
        high_end_visit_count = 0
        high_end_visit_rate = 0.0
    record.update(
        {
            "status": "failed",
            "failure": {
                "stage": "step",
                "type": "RuntimeError",
                "message": "synthetic post-step failure",
                "accepted_events": 1,
            },
            "outcome": None,
            "partial_outcome": {
                "summary_mode": "streaming_o1_no_retained_events",
                "configured_horizon": horizon,
                "accepted_events": 1,
                "reward_sum": 0.0,
                "mean_reward": 0.0,
                "oracle_reward_sum": oracle_reward,
                "regret_sum": oracle_reward,
                "parameter_change_events": 0,
                "phase_event_counts": [1, 0],
                "phase_reward_sums": [0.0, 0.0],
                "windows": windows,
                "high_end_visit_count": high_end_visit_count,
                "high_end_visit_rate": high_end_visit_rate,
            },
        }
    )
    record["telemetry"].update(
        {
            "warmed_step_seconds_total": 0.0,
            "warmed_step_count": 0,
            "warmed_step_seconds_mean": None,
        }
    )
    _redigest(record)

    with pytest.raises(ValueError, match="windows exceed"):
        scorecard.validate_scorecard_run_record(record)


def test_completed_shard_rejects_self_asserted_zero_resource_payload() -> None:
    plan = build_development_plan()
    spec = scorecard.iter_run_specs(plan)[0]
    record = _completed_record(plan, spec)
    for snapshot in record["outcome"]["resources"].values():
        snapshot["persistent_jax_array_leaves"] = 0
        snapshot["persistent_jax_array_scalar_count"] = 0
        snapshot["persistent_jax_array_bytes"] = 0
        snapshot["persistent_static_numeric_bytes"] = 0
        snapshot["persistent_numeric_bytes_total"] = 0
        snapshot["trainable_scalar_count_estimate"] = 0
        if "trainable_parameter_tree_scalar_count" in snapshot:
            snapshot["trainable_parameter_tree_scalar_count"] = 0
    _redigest(record)
    with pytest.raises(ValueError, match="canonical agent state"):
        scorecard.validate_scorecard_run_record(record)


def test_completed_oracle_shard_rejects_rehashed_swapped_payoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_development_plan()
    spec = next(
        item
        for item in scorecard.iter_run_specs(plan)
        if item.environment_kind == "switching_two_state"
        and item.arm == "privileged_oracle"
        and item.seed == SEED_ROSTER[0]
    )
    rogue_config = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=scorecard.SWITCHING_PHASE_LENGTH,
        payoffs_a=((1.0, 0.0), (0.0, 1.0)),
        payoffs_b=((0.0, 1.0), (1.0, 0.0)),
    )
    with monkeypatch.context() as context:
        context.setattr(scorecard, "_switching_environment_config", lambda _plan: rogue_config)
        rogue_runner = scorecard.build_scorecard_runner(plan, spec)
        record = _completed_record(plan, spec, runner=rogue_runner)

    with pytest.raises(ValueError, match="environment definition|canonical resolved"):
        scorecard.validate_scorecard_run_record(record)


@pytest.mark.parametrize(
    "tamper",
    (
        "telemetry_extra_field",
        "resource_extra_field",
        "window_extra_field",
        "window_mean",
        "fractional_resource_count",
        "resource_count_relationship",
        "telemetry_count",
        "oracle_sum",
        "inactive_phase_reward",
    ),
)
def test_completed_shard_rejects_nested_schema_and_numeric_tampering(
    tamper: str,
) -> None:
    plan = build_development_plan()
    spec = next(
        item
        for item in scorecard.iter_run_specs(plan)
        if item.environment_kind == "riverswim"
        and item.arm == "prototype"
        and item.seed == SEED_ROSTER[0]
    )
    record = _completed_record(plan, spec)
    outcome = record["outcome"]
    if tamper == "telemetry_extra_field":
        record["telemetry"]["selection_latency"] = 0.0
    elif tamper == "resource_extra_field":
        outcome["resources"]["initial"]["unbound_estimate"] = 0
    elif tamper == "window_extra_field":
        outcome["windows"]["early"]["unbound_metric"] = 0.0
    elif tamper == "window_mean":
        outcome["windows"]["early"]["mean_reward"] = 0.5
    elif tamper == "fractional_resource_count":
        outcome["resources"]["initial"]["persistent_jax_array_leaves"] = 1.5
    elif tamper == "resource_count_relationship":
        outcome["resources"]["initial"]["trainable_scalar_count_estimate"] = 2
    elif tamper == "telemetry_count":
        record["telemetry"]["warmed_step_count"] = 0
        record["telemetry"]["warmed_step_seconds_mean"] = None
    elif tamper == "oracle_sum":
        outcome["oracle_reward_sum"] += 1.0
        outcome["regret_sum"] += 1.0
    elif tamper == "inactive_phase_reward":
        outcome["phase_reward_sums"] = [-1.0, 1.0]
    else:
        raise AssertionError(f"unknown tamper case {tamper}")
    _redigest(record)

    with pytest.raises(ValueError):
        scorecard.validate_scorecard_run_record(record)


def _assert_all_gate_flags_fail(summary: dict[str, Any]) -> None:
    assert summary["control_calibration_gate_passed"] is False
    assert summary["candidate_utility_gate_passed"] is False
    for environment in ENVIRONMENT_ROSTER:
        environment_summary = summary["environments"][environment]
        assert environment_summary["control_calibration"]["qualified"] is False
        checks = environment_summary["candidate_development_checks"]
        assert checks["prototype_vs_frozen_passed"] is False
        assert checks["sarsa_noninferiority_passed"] is False
        assert checks["utility_gate_passed"] is False


def test_execution_failure_forces_every_summary_gate_to_fail() -> None:
    records = _summary_records()
    failed = next(record for record in records if record["arm"] == "prototype_frozen")
    failed["status"] = "failed"
    failed["outcome"] = None
    failed["failure"] = {"stage": "step", "type": "RuntimeError", "message": "boom"}

    summary = scorecard._summarize_validated_run_records(
        build_development_plan(), records
    )

    assert summary["status"] == "valid_execution_failure"
    _assert_all_gate_flags_fail(summary)


def test_parameter_change_failure_forces_every_summary_gate_to_fail() -> None:
    records = _summary_records()
    records[0]["outcome"]["parameter_change_check"] = {"passed": False}

    summary = scorecard._summarize_validated_run_records(
        build_development_plan(), records
    )

    assert summary["status"] == "valid_parameter_change_failure"
    _assert_all_gate_flags_fail(summary)


def test_strict_json_loader_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink|regular|strict JSON"):
        scorecard.load_json_strict(link)


def test_strict_json_loader_rejects_oversized_regular_files(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.touch()
    os.truncate(path, scorecard.MAX_SCORECARD_JSON_INPUT_BYTES + 1)

    with pytest.raises(ValueError, match="exceeds|too large"):
        scorecard.load_json_strict(path)


def test_strict_json_loader_rejects_nonregular_inputs_without_blocking(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.fifo"
    os.mkfifo(path)

    with pytest.raises(ValueError, match="regular file"):
        scorecard.load_json_strict(path)


def test_aggregate_rejects_wrong_path_count_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(path: Path) -> dict[str, Any]:
        raise AssertionError(f"unexpected load: {path}")

    monkeypatch.setattr(scorecard, "load_json_strict", unexpected_load)
    with pytest.raises(ValueError, match="exactly 144"):
        scorecard.summarize_shard_files([])


def test_aggregate_rejects_hard_links_using_opened_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "shard.json"
    shard.write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias.json"
    os.link(shard, alias)
    monkeypatch.setattr(
        scorecard,
        "validate_scorecard_run_record",
        lambda *args, **kwargs: {"valid": True},
    )
    paths = [shard, alias, *[shard] * 142]

    with pytest.raises(ValueError, match="unique regular files"):
        scorecard.summarize_shard_files(paths)


@pytest.fixture(scope="module")
def completed_artifact() -> dict[str, Any]:
    plan = build_development_plan()
    identities = scorecard._current_consistency_identities()
    records = [
        _completed_record(plan, spec, identities=identities)
        for spec in scorecard.iter_run_specs(plan)
    ]
    return scorecard.build_scorecard_artifact(plan, records)


def test_valid_completed_aggregate_recomputes_without_promotion(
    completed_artifact: dict[str, Any],
) -> None:
    result = scorecard.validate_scorecard_artifact(completed_artifact)

    assert result["valid"] is True
    assert result["permanently_nonpromoting"] is True
    assert completed_artifact["summary"]["cross_environment_pooled_score"] is None


def test_aggregate_rejects_boolean_schema_version_and_summary_integer(
    completed_artifact: dict[str, Any],
) -> None:
    for mutation in ("schema_version", "summary_run_count"):
        artifact = copy.deepcopy(completed_artifact)
        if mutation == "schema_version":
            artifact["schema_version"] = True
        else:
            artifact["summary"]["run_count"] = True
        artifact["artifact_sha256"] = scorecard._digest_excluding(
            artifact, "artifact_sha256"
        )
        with pytest.raises(ValueError):
            scorecard.validate_scorecard_artifact(artifact)


def test_failed_record_requires_exact_partial_schema_and_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"schema": "test.identity.v1", "value": "fixed"}
    monkeypatch.setattr(scorecard, "_checkpoint_source_identity", lambda: identity)
    monkeypatch.setattr(scorecard, "_checkpoint_runtime_identity", lambda: identity)
    monkeypatch.setattr(scorecard, "_checkpoint_dependency_identity", lambda: identity)
    monkeypatch.setattr(
        scorecard,
        "build_scorecard_runner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("build failed")),
    )
    plan = build_development_plan()
    record = scorecard.run_scorecard_shard(plan, scorecard.iter_run_specs(plan)[0])
    for mutation in ("extra_partial", "unknown_stage", "build_count"):
        altered = copy.deepcopy(record)
        if mutation == "extra_partial":
            altered["partial_outcome"]["extra"] = 1
        elif mutation == "unknown_stage":
            altered["failure"]["stage"] = "unknown"
        else:
            altered["failure"]["accepted_events"] = 1
            altered["partial_outcome"]["accepted_events"] = 1
        _redigest(altered)
        with pytest.raises(ValueError):
            scorecard.validate_scorecard_run_record(altered, plan=plan)


def test_run_shard_cli_validates_before_immutable_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_development_plan()
    spec = scorecard.iter_run_specs(plan)[0]
    malformed = _completed_record(plan, spec)
    malformed["schedule_index"] = True
    _redigest(malformed)
    monkeypatch.setattr(scorecard, "run_scorecard_shard", lambda _plan, _spec: malformed)
    output = tmp_path / "invalid-shard.json"
    with pytest.raises(ValueError, match="canonical cyclic schedule"):
        scorecard.main(
            [
                "run-shard",
                "--environment",
                spec.environment_kind,
                "--arm",
                spec.arm,
                "--seed",
                str(spec.seed),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize("kind", ("shard", "aggregate"))
def test_validate_cli_accepts_valid_completed_inputs(
    kind: str,
    completed_artifact: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if kind == "aggregate":
        payload = completed_artifact
    else:
        plan = build_development_plan()
        payload = _completed_record(plan, scorecard.iter_run_specs(plan)[0])
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert scorecard.main(["validate", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["permanently_nonpromoting"] is True


def test_validate_cli_accepts_the_canonical_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "plan.json"
    payload = scorecard.build_development_plan().to_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert scorecard.main(["validate", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["evidence_policy"]["permanently_nonpromoting"] is True
