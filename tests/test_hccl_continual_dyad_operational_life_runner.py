# mypy: disable-error-code="arg-type,untyped-decorator"
"""Cheap contracts for the private primitive operational-life runner."""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from alberta_framework.core.hccl_continual_dyad_operational_life_runner import (
    HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_CONFIG_SCHEMA,
    HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_EVIDENCE_LEVEL,
    HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_METADATA_SCHEMA,
    HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_STATUS,
    HCCLContinualDyadOperationalLifeError,
    HCCLContinualDyadOperationalLifeRunner,
    HCCLContinualDyadOperationalLifeRunnerConfig,
    _checkpoint_plan,
    _collect_operational_life,
)
from alberta_framework.core.hccl_continual_dyad_runner import (
    HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA,
    HCCLContinualDyadLifeTrace,
    validate_hccl_continual_dyad_life_trace,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_REGIME_NAMES,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCLCausalCoreFactors,
    hccl_causal_core_schedule_for_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]

_SCORES = np.asarray((0.125, 0.25, 0.5, 0.75), dtype=np.float32)


def _regime_for_step(profile: str, step: int) -> int:
    for name, start, end in hccl_causal_core_schedule_for_profile(profile):
        if start <= step < end:
            return HCCL_CAUSAL_CORE_REGIME_NAMES.index(name)
    raise AssertionError("fake step lies outside the fixed schedule")


def _state(words: tuple[int, int]) -> SimpleNamespace:
    return SimpleNamespace(
        hccl_state=SimpleNamespace(
            world_state=SimpleNamespace(
                step_words=np.asarray(words, dtype=np.uint32),
            )
        )
    )


def _score_for_regime(regime_id: int, factors: HCCLCausalCoreFactors) -> object:
    values = (
        factors.gathering,
        factors.velocity,
        factors.convention_clean,
        factors.convention_noisy,
    )
    return np.asarray(values[regime_id], dtype=np.float32)


class _FakeOperationalExecutor:
    """Structural compact-executor fake; no transaction or JAX event is run."""

    def __init__(
        self,
        config: HCCLContinualDyadOperationalLifeRunnerConfig,
        *,
        bad_clock_at: int | None = None,
        bad_checkpoint_report_at: int | None = None,
    ) -> None:
        self.config = config
        self.plan = _checkpoint_plan(config)
        self.bad_clock_at = bad_clock_at
        self.bad_checkpoint_report_at = bad_checkpoint_report_at
        self.initial_validation_count = 1
        self._absolute_step = 0
        self._state = _state((0, 0))
        self.embedded_checkpoint_events: list[int] = []
        self.explicit_checkpoint_events: list[int] = []
        self.step_calls: list[int] = []

    @property
    def state(self) -> object:
        return self._state

    @property
    def absolute_step(self) -> int:
        return self._absolute_step

    @property
    def checkpoint_interval(self) -> int:
        return 64

    def step(self, next_hard_action_masks: object) -> object:
        masks = np.asarray(next_hard_action_masks)
        assert masks.shape == (2, 2)
        assert masks.dtype == np.dtype(np.bool_)
        assert bool(np.all(masks))
        step = self._absolute_step
        self.step_calls.append(step)
        post = step + 1
        regime_id = _regime_for_step(self.config.schedule_profile, step)
        signals = SimpleNamespace(
            task_score=np.asarray(_SCORES[regime_id], dtype=np.float32),
            net_reward=np.full((2,), _SCORES[regime_id], dtype=np.float32),
        )
        factors = SimpleNamespace(
            gathering=np.asarray(_SCORES[0], dtype=np.float32),
            velocity=np.asarray(_SCORES[1], dtype=np.float32),
            convention_clean=np.asarray(_SCORES[2], dtype=np.float32),
            convention_noisy=np.asarray(_SCORES[3], dtype=np.float32),
        )
        proposal = SimpleNamespace(
            evaluator_regime_id=np.asarray(regime_id, dtype=np.int32),
            signals=signals,
            factors=factors,
        )
        transcript_post = post + 1 if self.bad_clock_at == step else post
        transcript = SimpleNamespace(
            pp_proposal=proposal,
            pre_transaction_words=np.asarray((0, step), dtype=np.uint32),
            post_transaction_words=np.asarray((0, transcript_post), dtype=np.uint32),
        )
        self._absolute_step = post
        self._state = _state((0, post))
        embedded = post in self.plan.executor_embedded_checkpoint_events
        if embedded:
            self.embedded_checkpoint_events.append(post)
        reported = int(embedded)
        if self.bad_checkpoint_report_at == step:
            reported = 1 - reported
        return SimpleNamespace(
            state=self._state,
            transcript=transcript,
            work=SimpleNamespace(runner_checkpoint_state_validations=reported),
            update_applied=True,
        )

    def validate_checkpoint(self) -> object:
        self.explicit_checkpoint_events.append(self._absolute_step)
        return self._state


@pytest.mark.parametrize(
    (
        "config",
        "profile",
        "steps",
        "cadence",
        "boundaries",
        "embedded",
        "explicit",
        "deduplicated",
        "duplicates",
    ),
    (
        (
            HCCLContinualDyadOperationalLifeRunnerConfig(),
            HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
            8_998,
            140,
            9,
            141,
            9,
            150,
            0,
        ),
        (
            HCCLContinualDyadOperationalLifeRunnerConfig.mechanics_smoke(),
            HCCL_CAUSAL_CORE_SMOKE_PROFILE,
            420,
            6,
            9,
            7,
            8,
            15,
            1,
        ),
        (
            HCCLContinualDyadOperationalLifeRunnerConfig.core_l2(),
            HCCL_CAUSAL_CORE_L2_PROFILE,
            71_984,
            1_124,
            79,
            1_125,
            77,
            1_202,
            2,
        ),
        (
            HCCLContinualDyadOperationalLifeRunnerConfig.core_l3(),
            HCCL_CAUSAL_CORE_L3_PROFILE,
            1_007_776,
            15_746,
            1_119,
            15_747,
            1_101,
            16_848,
            18,
        ),
    ),
)
def test_four_profiles_have_exact_deduplicated_checkpoint_plans(
    config: HCCLContinualDyadOperationalLifeRunnerConfig,
    profile: str,
    steps: int,
    cadence: int,
    boundaries: int,
    embedded: int,
    explicit: int,
    deduplicated: int,
    duplicates: int,
) -> None:
    plan = _checkpoint_plan(config)
    assert config.schedule_profile == profile
    assert config.total_steps == steps
    assert len(plan.feature_cadence_events) == cadence
    assert len(plan.internal_segment_boundary_events) == boundaries
    assert len(plan.executor_embedded_checkpoint_events) == embedded
    assert len(plan.explicit_boundary_checkpoint_events) == explicit
    assert len(plan.deduplicated_checkpoint_events) == deduplicated
    assert plan.duplicate_checkpoint_triggers_suppressed == duplicates
    assert plan.final_event == steps
    assert plan.deduplicated_checkpoint_events[-1] == steps
    assert set(plan.explicit_boundary_checkpoint_events).isdisjoint(
        plan.executor_embedded_checkpoint_events
    )

    payload = config.to_config()
    assert payload["schema"] == HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_CONFIG_SCHEMA
    assert payload["trace_schema"] == HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA
    assert payload["status"] == HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_STATUS
    assert payload["evidence_level"] == HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_EVIDENCE_LEVEL
    assert payload["feature_lifecycle_checkpoint_cadence"] == 64
    assert payload["deduplicated_full_checkpoint_count"] == deduplicated
    assert payload["initial_validation_executor_owned"] is True
    assert payload["persistent_state_and_transcript_equivalence_pending"] is True
    for name in (
        "protocol_seed_reservation_or_consumption_authorized",
        "benchmark_execution_authorized",
        "output_writes_authorized",
        "artifact_authorized",
        "threshold_authorized",
        "evidence_authorized",
        "promotion_authorized",
        "scientific_promotion_allowed",
    ):
        assert payload[name] is False
    assert HCCLContinualDyadOperationalLifeRunnerConfig.from_config(payload) == config
    assert HCCLContinualDyadOperationalLifeRunnerConfig.from_json(config.to_json()) == config


def test_config_json_is_strict_and_noncanonical_authority_is_rejected() -> None:
    config = HCCLContinualDyadOperationalLifeRunnerConfig.mechanics_smoke()
    changed = config.to_config()
    changed["evidence_authorized"] = True
    with pytest.raises(ValueError, match="noncanonical"):
        HCCLContinualDyadOperationalLifeRunnerConfig.from_config(changed)
    with pytest.raises(ValueError, match="non-strict"):
        HCCLContinualDyadOperationalLifeRunnerConfig.from_json(
            '{"schedule_profile":"x","schedule_profile":"y"}'
        )
    with pytest.raises(ValueError, match="non-strict"):
        HCCLContinualDyadOperationalLifeRunnerConfig.from_json('{"value":NaN}')
    with pytest.raises(ValueError, match="one object"):
        HCCLContinualDyadOperationalLifeRunnerConfig.from_json("[]")


def test_fake_smoke_life_collects_exact_existing_trace_and_checkpoint_metadata() -> None:
    config = HCCLContinualDyadOperationalLifeRunnerConfig.mechanics_smoke()
    executor = _FakeOperationalExecutor(config)
    collected = _collect_operational_life(config, executor, _score_for_regime)
    trace = validate_hccl_continual_dyad_life_trace(collected.trace)
    plan = _checkpoint_plan(config)

    assert type(trace) is HCCLContinualDyadLifeTrace
    assert trace.schema == HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA
    assert trace.total_steps == 420
    np.testing.assert_array_equal(
        trace.all_regime_score_matrix,
        np.broadcast_to(_SCORES, (420, 4)),
    )
    np.testing.assert_array_equal(
        trace.task_scores,
        trace.all_regime_score_matrix[np.arange(420), trace.regime_ids],
    )
    assert executor.initial_validation_count == 1
    assert executor.step_calls == list(range(420))
    assert tuple(executor.embedded_checkpoint_events) == (
        plan.executor_embedded_checkpoint_events
    )
    assert tuple(executor.explicit_checkpoint_events) == (
        plan.explicit_boundary_checkpoint_events
    )
    assert collected.state is executor.state

    metadata = collected.metadata
    assert metadata.schema == HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_METADATA_SCHEMA
    assert metadata.operational_executor_step_calls == 420
    assert metadata.operational_results_committed == 420
    assert metadata.evaluator_readouts == 420
    assert metadata.initial_executor_state_validations == 1
    assert metadata.executor_embedded_checkpoint_validations == 7
    assert metadata.explicit_boundary_checkpoint_validations == 8
    assert metadata.deduplicated_full_checkpoint_validations == 15
    assert metadata.total_full_state_validations == 16
    assert metadata.duplicate_checkpoint_triggers_suppressed == 1
    assert metadata.output_write_calls == 0
    assert metadata.artifact_bytes_written == 0
    assert metadata.persistent_state_and_transcript_equivalence_pending is True
    assert metadata.scientific_promotion_allowed is False


@pytest.mark.parametrize(
    ("bad_clock_at", "bad_checkpoint_report_at", "stage"),
    (
        (3, None, "event-clock"),
        (None, 0, "checkpoint-accounting"),
    ),
)
def test_fake_collection_fails_closed_on_clock_or_checkpoint_mismatch(
    bad_clock_at: int | None,
    bad_checkpoint_report_at: int | None,
    stage: str,
) -> None:
    config = HCCLContinualDyadOperationalLifeRunnerConfig.mechanics_smoke()
    executor = _FakeOperationalExecutor(
        config,
        bad_clock_at=bad_clock_at,
        bad_checkpoint_report_at=bad_checkpoint_report_at,
    )
    with pytest.raises(HCCLContinualDyadOperationalLifeError) as captured:
        _collect_operational_life(config, executor, _score_for_regime)
    assert captured.value.stage == stage
    assert not hasattr(captured.value, "trace")


def test_metadata_is_strict_and_runner_construction_does_not_execute() -> None:
    config = HCCLContinualDyadOperationalLifeRunnerConfig.mechanics_smoke()
    executor = _FakeOperationalExecutor(config)
    metadata = _collect_operational_life(config, executor, _score_for_regime).metadata
    assert metadata.to_config()["status"] == HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_STATUS
    with pytest.raises(ValueError, match="checkpoint"):
        dataclasses.replace(
            metadata,
            deduplicated_full_checkpoint_validations=(
                metadata.deduplicated_full_checkpoint_validations - 1
            ),
        )

    runner = HCCLContinualDyadOperationalLifeRunner(config)
    assert runner.config is config
    assert runner.to_config() == config.to_config()
    source = inspect.getsource(HCCLContinualDyadOperationalLifeRunner.run)
    assert "_HCCLContinualDyadOperationalExecutor" in source
    assert "checkpoint_interval=_FEATURE_LIFECYCLE_CHECKPOINT_CADENCE" in source
    assert ".state_valid(" not in source


def test_module_exports_no_writer_cli_evidence_or_promotion_surface() -> None:
    import alberta_framework.core.hccl_continual_dyad_operational_life_runner as module

    exports = cast(tuple[str, ...], module.__all__)
    assert "HCCLContinualDyadOperationalLifeRunner" in exports
    assert "HCCLContinualDyadOperationalLifeRunnerConfig" in exports
    assert "HCCLContinualDyadOperationalLifeResult" in exports
    assert not hasattr(module, "main")
    assert not hasattr(HCCLContinualDyadOperationalLifeRunner, "write")
    assert not hasattr(HCCLContinualDyadOperationalLifeRunner, "promote")
