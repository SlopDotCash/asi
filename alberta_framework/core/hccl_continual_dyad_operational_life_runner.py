# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return"
"""Private bounded-life owner for the primitive HCCL operational executor.

This additive L0 runner initializes through the canonical four-profile factory
and advances only through the compact operational executor.  The executor owns
the initial exact validation, every 64-event feature-lifecycle checkpoint, and
the final checkpoint.  This layer adds exact checks at internal schedule
segment boundaries that are not already covered and deduplicates all triggers.

Evaluator-only regime scores are derived after each committed operational
result from the PP factors retained by its transcript.  Evaluator labels and
counterfactual score columns are never passed back to a learner.  Persistent
state/transcript equivalence with the audited transaction remains pending a
real differential run; this module grants no output, artifact, seed protocol,
benchmark, evidence, threshold, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework.core.hccl_continual_dyad_factory import (
    HCCLContinualDyadFactory,
    HCCLContinualDyadFactoryConfig,
    HCCLContinualDyadFactoryInitialization,
)
from alberta_framework.core.hccl_continual_dyad_operational_runner import (
    _HCCLContinualDyadOperationalExecutor,
)
from alberta_framework.core.hccl_continual_dyad_runner import (
    HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA,
    HCCLContinualDyadLifeTrace,
    validate_hccl_continual_dyad_life_trace,
)
from alberta_framework.core.hccl_continual_dyad_transaction import (
    HCCLContinualDyadState,
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

HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_CONFIG_SCHEMA: Final = (
    "alberta.hccl-continual-dyad-operational-life.config.v1"
)
HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_METADATA_SCHEMA: Final = (
    "alberta.hccl-continual-dyad-operational-life.metadata.v1"
)
HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_STATUS: Final = (
    "l0-development-private-primitive-operational-complete-life"
)
HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_EVIDENCE_LEVEL: Final = "L0"
HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_LIMITATIONS: Final = (
    "private-primitive-operational-executor-only",
    "host-eager-only",
    "persistent-state-and-transcript-equivalence-pending-differential-run",
    "complete-in-memory-trace-only",
    "caller-key-material-is-not-a-reserved-consumed-or-held-out-seed",
    "no-output-artifact-threshold-benchmark-evidence-or-promotion-authority",
)

_N_AGENTS: Final = 2
_N_ACTIONS: Final = 2
_N_REGIMES: Final = 4
_FEATURE_LIFECYCLE_CHECKPOINT_CADENCE: Final = 64
_SUPPORTED_PROFILES: Final = (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
)

type Float32Array = NDArray[np.float32]
type Int32Array = NDArray[np.int32]
type UInt32Array = NDArray[np.uint32]
type BoolArray = NDArray[np.bool_]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadOperationalLifeRunnerConfig:
    """Select one factory-owned complete life for compact execution."""

    factory_config: HCCLContinualDyadFactoryConfig = dataclasses.field(
        default_factory=HCCLContinualDyadFactoryConfig
    )

    def __post_init__(self) -> None:
        if type(self.factory_config) is not HCCLContinualDyadFactoryConfig:
            raise TypeError("factory_config must be exact HCCLContinualDyadFactoryConfig")
        if self.factory_config.schedule_profile not in _SUPPORTED_PROFILES:
            raise ValueError("factory config does not select a supported complete life")

    @classmethod
    def mechanics_smoke(cls) -> HCCLContinualDyadOperationalLifeRunnerConfig:
        return cls(factory_config=HCCLContinualDyadFactoryConfig.mechanics_smoke())

    @classmethod
    def core_l2(cls) -> HCCLContinualDyadOperationalLifeRunnerConfig:
        return cls(factory_config=HCCLContinualDyadFactoryConfig.core_l2())

    @classmethod
    def core_l3(cls) -> HCCLContinualDyadOperationalLifeRunnerConfig:
        return cls(factory_config=HCCLContinualDyadFactoryConfig.core_l3())

    @property
    def schedule_profile(self) -> str:
        return self.factory_config.schedule_profile

    @property
    def total_steps(self) -> int:
        return self.factory_config.maximum_committed_transitions

    def to_config(self) -> dict[str, object]:
        plan = _checkpoint_plan(self)
        return {
            "type": type(self).__name__,
            "schema": HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_CONFIG_SCHEMA,
            "trace_schema": HCCL_CONTINUAL_DYAD_LIFE_TRACE_SCHEMA,
            "metadata_schema": HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_METADATA_SCHEMA,
            "status": HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_STATUS,
            "evidence_level": HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_EVIDENCE_LEVEL,
            "factory_config": self.factory_config.to_config(),
            "schedule_profile": self.schedule_profile,
            "total_steps": self.total_steps,
            "complete_fixed_life_only": True,
            "partial_life_supported": False,
            "fresh_factory_initialization_per_run": True,
            "compact_operational_executor_only": True,
            "initial_validation_executor_owned": True,
            "feature_lifecycle_checkpoint_cadence": (
                _FEATURE_LIFECYCLE_CHECKPOINT_CADENCE
            ),
            "feature_lifecycle_checkpoint_trigger_count": len(
                plan.feature_cadence_events
            ),
            "internal_segment_boundary_checkpoint_trigger_count": len(
                plan.internal_segment_boundary_events
            ),
            "final_checkpoint_trigger_count": 1,
            "executor_embedded_checkpoint_count": len(
                plan.executor_embedded_checkpoint_events
            ),
            "explicit_boundary_checkpoint_count": len(
                plan.explicit_boundary_checkpoint_events
            ),
            "deduplicated_full_checkpoint_count": len(
                plan.deduplicated_checkpoint_events
            ),
            "duplicate_checkpoint_triggers_suppressed": (
                plan.duplicate_checkpoint_triggers_suppressed
            ),
            "reset_callback_count": 0,
            "boundary_callback_count": 0,
            "evaluator_columns_computed_after_operational_commit": True,
            "evaluator_labels_exposed_to_learner": False,
            "counterfactual_score_columns_exposed_to_learner": False,
            "persistent_state_and_transcript_equivalence_pending": True,
            "bounded_development_life_execution_authorized": True,
            "caller_key_material_used": True,
            "protocol_seed_reservation_or_consumption_authorized": False,
            "benchmark_execution_authorized": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "threshold_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "scientific_promotion_allowed": False,
            "artifact_write_calls": 0,
            "artifact_bytes_written": 0,
            "limitations": list(HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_LIMITATIONS),
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLContinualDyadOperationalLifeRunnerConfig:
        if type(payload) is not dict:
            raise TypeError("operational-life config must be an exact dict")
        nested = payload.get("factory_config")
        if type(nested) is not dict:
            raise ValueError("operational-life factory_config must be an exact dict")
        candidate = cls(
            factory_config=HCCLContinualDyadFactoryConfig.from_config(nested)
        )
        if _canonical_json_bytes(payload) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("operational-life config is noncanonical or unsupported")
        return candidate

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_config()).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> HCCLContinualDyadOperationalLifeRunnerConfig:
        if type(payload) is not str:
            raise TypeError("operational-life JSON must be an exact string")
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("operational-life JSON is invalid or non-strict") from error
        if type(decoded) is not dict:
            raise ValueError("operational-life JSON must encode one object")
        return cls.from_config(decoded)


@dataclasses.dataclass(frozen=True, slots=True)
class _OperationalCheckpointPlan:
    feature_cadence_events: tuple[int, ...]
    internal_segment_boundary_events: tuple[int, ...]
    final_event: int
    executor_embedded_checkpoint_events: tuple[int, ...]
    explicit_boundary_checkpoint_events: tuple[int, ...]
    deduplicated_checkpoint_events: tuple[int, ...]
    duplicate_checkpoint_triggers_suppressed: int

    def __post_init__(self) -> None:
        for name in (
            "feature_cadence_events",
            "internal_segment_boundary_events",
            "executor_embedded_checkpoint_events",
            "explicit_boundary_checkpoint_events",
            "deduplicated_checkpoint_events",
        ):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not int for item in value):
                raise TypeError(f"{name} must be an exact integer tuple")
            if tuple(sorted(set(value))) != value:
                raise ValueError(f"{name} must be sorted and duplicate-free")
        if type(self.final_event) is not int or self.final_event < 1:
            raise ValueError("final_event must be a positive exact int")
        if type(self.duplicate_checkpoint_triggers_suppressed) is not int:
            raise TypeError("duplicate checkpoint count must be an exact int")
        if self.duplicate_checkpoint_triggers_suppressed < 0:
            raise ValueError("duplicate checkpoint count cannot be negative")
        embedded = set(self.feature_cadence_events) | {self.final_event}
        explicit = set(self.internal_segment_boundary_events) - embedded
        all_events = embedded | set(self.internal_segment_boundary_events)
        if (
            self.executor_embedded_checkpoint_events != tuple(sorted(embedded))
            or self.explicit_boundary_checkpoint_events != tuple(sorted(explicit))
            or self.deduplicated_checkpoint_events != tuple(sorted(all_events))
            or self.deduplicated_checkpoint_events[-1] != self.final_event
        ):
            raise ValueError("operational checkpoint plan is not the exact deduplicated union")
        trigger_count = (
            len(self.feature_cadence_events)
            + len(self.internal_segment_boundary_events)
            + 1
        )
        if (
            trigger_count - len(self.deduplicated_checkpoint_events)
            != self.duplicate_checkpoint_triggers_suppressed
        ):
            raise ValueError("operational checkpoint deduplication count is inconsistent")


def _checkpoint_plan(
    config: HCCLContinualDyadOperationalLifeRunnerConfig,
) -> _OperationalCheckpointPlan:
    if type(config) is not HCCLContinualDyadOperationalLifeRunnerConfig:
        raise TypeError("config must be exact operational-life config")
    total = config.total_steps
    schedule = hccl_causal_core_schedule_for_profile(config.schedule_profile)
    cursor = 0
    internal_boundaries: list[int] = []
    for _name, start, end in schedule:
        if start != cursor or not start < end or end > total:
            raise ValueError("fixed HCCL schedule segments are not contiguous and bounded")
        if end < total:
            internal_boundaries.append(end)
        cursor = end
    if cursor != total:
        raise ValueError("fixed HCCL schedule does not close the complete life")
    cadence = tuple(
        range(
            _FEATURE_LIFECYCLE_CHECKPOINT_CADENCE,
            total + 1,
            _FEATURE_LIFECYCLE_CHECKPOINT_CADENCE,
        )
    )
    boundaries = tuple(internal_boundaries)
    embedded = tuple(sorted(set(cadence) | {total}))
    explicit = tuple(sorted(set(boundaries) - set(embedded)))
    deduplicated = tuple(sorted(set(embedded) | set(boundaries)))
    duplicate_count = len(cadence) + len(boundaries) + 1 - len(deduplicated)
    return _OperationalCheckpointPlan(
        feature_cadence_events=cadence,
        internal_segment_boundary_events=boundaries,
        final_event=total,
        executor_embedded_checkpoint_events=embedded,
        explicit_boundary_checkpoint_events=explicit,
        deduplicated_checkpoint_events=deduplicated,
        duplicate_checkpoint_triggers_suppressed=duplicate_count,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadOperationalLifeMetadata:
    """Truthful completed-life operational and checkpoint accounting."""

    schema: str
    status: str
    evidence_level: str
    schedule_profile: str
    total_events: int
    operational_executor_step_calls: int
    operational_results_committed: int
    evaluator_readouts: int
    initial_executor_state_validations: int
    feature_lifecycle_checkpoint_cadence: int
    feature_lifecycle_checkpoint_triggers: int
    internal_segment_boundary_checkpoint_triggers: int
    final_checkpoint_triggers: int
    executor_embedded_checkpoint_validations: int
    explicit_boundary_checkpoint_validations: int
    deduplicated_full_checkpoint_validations: int
    duplicate_checkpoint_triggers_suppressed: int
    total_full_state_validations: int
    reset_callback_count: int
    boundary_callback_count: int
    output_write_calls: int
    artifact_bytes_written: int
    persistent_state_and_transcript_equivalence_pending: bool
    scientific_promotion_allowed: bool

    def __post_init__(self) -> None:
        if self.schema != HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_METADATA_SCHEMA:
            raise ValueError("operational-life metadata schema is unsupported")
        if self.status != HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_STATUS:
            raise ValueError("operational-life metadata status is unsupported")
        if self.evidence_level != HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_EVIDENCE_LEVEL:
            raise ValueError("operational-life metadata evidence level is unsupported")
        scalar_names = (
            "total_events",
            "operational_executor_step_calls",
            "operational_results_committed",
            "evaluator_readouts",
            "initial_executor_state_validations",
            "feature_lifecycle_checkpoint_cadence",
            "feature_lifecycle_checkpoint_triggers",
            "internal_segment_boundary_checkpoint_triggers",
            "final_checkpoint_triggers",
            "executor_embedded_checkpoint_validations",
            "explicit_boundary_checkpoint_validations",
            "deduplicated_full_checkpoint_validations",
            "duplicate_checkpoint_triggers_suppressed",
            "total_full_state_validations",
            "reset_callback_count",
            "boundary_callback_count",
            "output_write_calls",
            "artifact_bytes_written",
        )
        for name in scalar_names:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"metadata {name} must be a nonnegative exact int")
        if type(self.schedule_profile) is not str:
            raise TypeError("metadata schedule_profile must be an exact string")
        config = HCCLContinualDyadOperationalLifeRunnerConfig(
            factory_config=HCCLContinualDyadFactoryConfig(
                schedule_profile=self.schedule_profile
            )
        )
        plan = _checkpoint_plan(config)
        expected = {
            "total_events": config.total_steps,
            "operational_executor_step_calls": config.total_steps,
            "operational_results_committed": config.total_steps,
            "evaluator_readouts": config.total_steps,
            "initial_executor_state_validations": 1,
            "feature_lifecycle_checkpoint_cadence": (
                _FEATURE_LIFECYCLE_CHECKPOINT_CADENCE
            ),
            "feature_lifecycle_checkpoint_triggers": len(plan.feature_cadence_events),
            "internal_segment_boundary_checkpoint_triggers": len(
                plan.internal_segment_boundary_events
            ),
            "final_checkpoint_triggers": 1,
            "executor_embedded_checkpoint_validations": len(
                plan.executor_embedded_checkpoint_events
            ),
            "explicit_boundary_checkpoint_validations": len(
                plan.explicit_boundary_checkpoint_events
            ),
            "deduplicated_full_checkpoint_validations": len(
                plan.deduplicated_checkpoint_events
            ),
            "duplicate_checkpoint_triggers_suppressed": (
                plan.duplicate_checkpoint_triggers_suppressed
            ),
            "total_full_state_validations": 1
            + len(plan.deduplicated_checkpoint_events),
            "reset_callback_count": 0,
            "boundary_callback_count": 0,
            "output_write_calls": 0,
            "artifact_bytes_written": 0,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"metadata {name} violates the exact checkpoint contract")
        if (
            type(self.persistent_state_and_transcript_equivalence_pending) is not bool
            or not self.persistent_state_and_transcript_equivalence_pending
        ):
            raise ValueError("operational equivalence must remain explicitly pending")
        if (
            type(self.scientific_promotion_allowed) is not bool
            or self.scientific_promotion_allowed
        ):
            raise ValueError("operational-life metadata cannot authorize promotion")

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


class HCCLContinualDyadOperationalLifeError(RuntimeError):
    """Fail-closed complete-life error with no partial trace."""

    def __init__(self, step_index: int, stage: str, detail: str) -> None:
        self.step_index = step_index
        self.stage = stage
        super().__init__(
            f"primitive operational life aborted at step {step_index} "
            f"during {stage}: {detail}"
        )


class _OperationalExecutor(Protocol):
    @property
    def state(self) -> object: ...

    @property
    def absolute_step(self) -> int: ...

    @property
    def checkpoint_interval(self) -> int | None: ...

    def step(self, next_hard_action_masks: Array) -> object: ...

    def validate_checkpoint(self) -> object: ...


type _TaskScoreForRegime = Callable[[int, HCCLCausalCoreFactors], object]


@dataclasses.dataclass(frozen=True, slots=True)
class _CollectedOperationalLife:
    trace: HCCLContinualDyadLifeTrace
    metadata: HCCLContinualDyadOperationalLifeMetadata
    state: object


def _host_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    try:
        result = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a host-convertible array") from error
    if result.shape != shape or result.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    return result


def _state_clock(state: object, *, name: str) -> UInt32Array:
    try:
        words = state.hccl_state.world_state.step_words
    except AttributeError as error:
        raise ValueError(f"{name} does not expose the world step words") from error
    return cast(
        UInt32Array,
        _host_array(
            words,
            name=f"{name}.hccl_state.world_state.step_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        ),
    )


def _exact_checkpoint_count(value: object) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError("operational result checkpoint count must be exact zero or one")
    return value


def _result_committed(value: object) -> bool:
    if type(value) is bool:
        committed = value
    else:
        committed = bool(
            _host_array(
                value,
                name="operational result update_applied",
                shape=(),
                dtype=np.dtype(np.bool_),
            )
        )
    if not committed:
        raise ValueError("operational result did not commit its candidate")
    return True


def _read_operational_event(
    config: HCCLContinualDyadOperationalLifeRunnerConfig,
    result: object,
    step_index: int,
    task_score_for_regime: _TaskScoreForRegime,
) -> tuple[int, np.float32, Float32Array, Float32Array]:
    try:
        proposal = result.transcript.pp_proposal
        signals = proposal.signals
        factor_source = proposal.factors
    except AttributeError as error:
        raise ValueError(
            "operational result does not expose the PP transcript interface"
        ) from error
    regime = _host_array(
        proposal.evaluator_regime_id,
        name="operational PP evaluator_regime_id",
        shape=(),
        dtype=np.dtype(np.int32),
    )
    regime_id = int(regime.item())
    if not 0 <= regime_id < _N_REGIMES:
        raise ValueError("operational PP evaluator regime lies outside the fixed domain")
    expected_regime = -1
    for name, start, end in hccl_causal_core_schedule_for_profile(
        config.schedule_profile
    ):
        if start <= step_index < end:
            expected_regime = HCCL_CAUSAL_CORE_REGIME_NAMES.index(name)
            break
    if regime_id != expected_regime:
        raise ValueError("operational PP evaluator regime differs from the fixed schedule")
    task = _host_array(
        signals.task_score,
        name="operational PP task_score",
        shape=(),
        dtype=np.dtype(np.float32),
    )
    net_rewards = cast(
        Float32Array,
        _host_array(
            signals.net_reward,
            name="operational PP net_reward",
            shape=(_N_AGENTS,),
            dtype=np.dtype(np.float32),
        ),
    )
    factor_values: list[object] = []
    for name in (
        "gathering",
        "velocity",
        "convention_clean",
        "convention_noisy",
    ):
        value = getattr(factor_source, name)
        host = _host_array(
            value,
            name=f"operational PP factors.{name}",
            shape=(),
            dtype=np.dtype(np.float32),
        )
        if not bool(np.isfinite(host)):
            raise ValueError(f"operational PP factor {name} is nonfinite")
        factor_values.append(value)
    factors = HCCLCausalCoreFactors(
        gathering=factor_values[0],
        velocity=factor_values[1],
        convention_clean=factor_values[2],
        convention_noisy=factor_values[3],
    )
    all_scores = np.asarray(
        tuple(
            np.float32(
                _host_array(
                    task_score_for_regime(index, factors),
                    name=f"evaluator score[{index}]",
                    shape=(),
                    dtype=np.dtype(np.float32),
                ).item()
            )
            for index in range(_N_REGIMES)
        ),
        dtype=np.float32,
    )
    task_score = np.float32(task.item())
    if not (
        bool(np.all(np.isfinite(all_scores)))
        and bool(np.isfinite(task_score))
        and bool(np.all(np.isfinite(net_rewards)))
    ):
        raise ValueError("operational evaluator readout is nonfinite")
    if all_scores[regime_id] != task_score:
        raise ValueError("selected evaluator score differs from operational PP task score")
    if not np.array_equal(
        net_rewards,
        np.full((_N_AGENTS,), task_score, dtype=np.float32),
    ):
        raise ValueError("operational PP net rewards differ from task score")
    return regime_id, task_score, net_rewards, all_scores


def _collect_operational_life(
    config: HCCLContinualDyadOperationalLifeRunnerConfig,
    executor: _OperationalExecutor,
    task_score_for_regime: _TaskScoreForRegime,
) -> _CollectedOperationalLife:
    """CI-cheap dependency-injection seam over one compact executor interface."""

    if type(config) is not HCCLContinualDyadOperationalLifeRunnerConfig:
        raise TypeError("config must be exact operational-life config")
    if not callable(task_score_for_regime):
        raise TypeError("task_score_for_regime must be callable")
    plan = _checkpoint_plan(config)
    if executor.checkpoint_interval != _FEATURE_LIFECYCLE_CHECKPOINT_CADENCE:
        raise ValueError("operational executor checkpoint interval must equal 64")
    if type(executor.absolute_step) is not int or executor.absolute_step != 0:
        raise ValueError("operational executor must begin at exact absolute step zero")
    if not np.array_equal(
        _state_clock(executor.state, name="initial executor state"),
        np.zeros((2,), dtype=np.uint32),
    ):
        raise ValueError("operational executor must begin at the exact zero world clock")

    steps = config.total_steps
    regime_ids = np.empty((steps,), dtype=np.int32)
    committed = np.empty((steps,), dtype=np.bool_)
    pre_words = np.empty((steps, 2), dtype=np.uint32)
    post_words = np.empty((steps, 2), dtype=np.uint32)
    task_scores = np.empty((steps,), dtype=np.float32)
    net_rewards = np.empty((steps, _N_AGENTS), dtype=np.float32)
    all_regime_scores = np.empty((steps, _N_REGIMES), dtype=np.float32)
    next_masks = jnp.ones((_N_AGENTS, _N_ACTIONS), dtype=jnp.bool_)
    embedded_validations = 0
    explicit_validations = 0
    embedded_events = set(plan.executor_embedded_checkpoint_events)
    explicit_events = set(plan.explicit_boundary_checkpoint_events)

    for step_index in range(steps):
        expected_pre = np.asarray((0, step_index), dtype=np.uint32)
        expected_post = np.asarray((0, step_index + 1), dtype=np.uint32)
        if type(executor.absolute_step) is not int or executor.absolute_step != step_index:
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "source-clock",
                "executor absolute step is discontinuous",
            )
        source = executor.state
        if not np.array_equal(
            _state_clock(source, name="executor source"),
            expected_pre,
        ):
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "source-clock",
                "executor source world clock is discontinuous",
            )
        try:
            result = executor.step(next_masks)
        except HCCLContinualDyadOperationalLifeError:
            raise
        except Exception as error:
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "operational-executor-step",
                str(error),
            ) from error
        try:
            result_state = result.state
            transcript = result.transcript
            work = result.work
            _result_committed(result.update_applied)
        except (AttributeError, TypeError, ValueError) as error:
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "result-contract",
                str(error),
            ) from error
        if (
            executor.absolute_step != step_index + 1
            or executor.state is not result_state
        ):
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "publication-binding",
                "executor did not publish the exact returned destination once",
            )
        try:
            event_pre = _host_array(
                transcript.pre_transaction_words,
                name="operational transcript pre_transaction_words",
                shape=(2,),
                dtype=np.dtype(np.uint32),
            )
            event_post = _host_array(
                transcript.post_transaction_words,
                name="operational transcript post_transaction_words",
                shape=(2,),
                dtype=np.dtype(np.uint32),
            )
            destination = _state_clock(result_state, name="operational destination")
        except (AttributeError, TypeError, ValueError) as error:
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "event-clock",
                str(error),
            ) from error
        if not (
            np.array_equal(event_pre, expected_pre)
            and np.array_equal(event_post, expected_post)
            and np.array_equal(destination, expected_post)
        ):
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "event-clock",
                "operational source, transcript, and destination clocks differ",
            )
        try:
            reported_checkpoints = _exact_checkpoint_count(
                work.runner_checkpoint_state_validations
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "checkpoint-accounting",
                str(error),
            ) from error
        expected_embedded = int(step_index + 1 in embedded_events)
        if reported_checkpoints != expected_embedded:
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "checkpoint-accounting",
                "executor checkpoint count differs from the deduplicated plan",
            )
        embedded_validations += reported_checkpoints
        if step_index + 1 in explicit_events:
            try:
                checkpointed = executor.validate_checkpoint()
            except Exception as error:
                raise HCCLContinualDyadOperationalLifeError(
                    step_index,
                    "segment-boundary-checkpoint",
                    str(error),
                ) from error
            if checkpointed is not result_state or executor.state is not result_state:
                raise HCCLContinualDyadOperationalLifeError(
                    step_index,
                    "segment-boundary-checkpoint",
                    "explicit checkpoint did not retain the exact destination",
                )
            explicit_validations += 1
        try:
            regime_id, task_score, event_net, scores = _read_operational_event(
                config,
                result,
                step_index,
                task_score_for_regime,
            )
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise HCCLContinualDyadOperationalLifeError(
                step_index,
                "post-commit-evaluator-readout",
                str(error),
            ) from error
        regime_ids[step_index] = regime_id
        committed[step_index] = True
        pre_words[step_index] = event_pre
        post_words[step_index] = event_post
        task_scores[step_index] = task_score
        net_rewards[step_index] = event_net
        all_regime_scores[step_index] = scores

    if (
        embedded_validations != len(plan.executor_embedded_checkpoint_events)
        or explicit_validations != len(plan.explicit_boundary_checkpoint_events)
    ):
        raise HCCLContinualDyadOperationalLifeError(
            steps,
            "checkpoint-accounting",
            "completed checkpoint counts differ from the exact life plan",
        )
    try:
        trace = HCCLContinualDyadLifeTrace(
            schedule_profile=config.schedule_profile,
            regime_ids=regime_ids,
            transaction_committed=committed,
            pre_step_words=pre_words,
            post_step_words=post_words,
            task_scores=task_scores,
            net_rewards=net_rewards,
            all_regime_score_matrix=all_regime_scores,
        )
    except (TypeError, ValueError) as error:
        raise HCCLContinualDyadOperationalLifeError(
            steps,
            "trace-finalization",
            str(error),
        ) from error
    metadata = HCCLContinualDyadOperationalLifeMetadata(
        schema=HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_METADATA_SCHEMA,
        status=HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_STATUS,
        evidence_level=HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_EVIDENCE_LEVEL,
        schedule_profile=config.schedule_profile,
        total_events=steps,
        operational_executor_step_calls=steps,
        operational_results_committed=steps,
        evaluator_readouts=steps,
        initial_executor_state_validations=1,
        feature_lifecycle_checkpoint_cadence=(
            _FEATURE_LIFECYCLE_CHECKPOINT_CADENCE
        ),
        feature_lifecycle_checkpoint_triggers=len(plan.feature_cadence_events),
        internal_segment_boundary_checkpoint_triggers=len(
            plan.internal_segment_boundary_events
        ),
        final_checkpoint_triggers=1,
        executor_embedded_checkpoint_validations=embedded_validations,
        explicit_boundary_checkpoint_validations=explicit_validations,
        deduplicated_full_checkpoint_validations=(
            embedded_validations + explicit_validations
        ),
        duplicate_checkpoint_triggers_suppressed=(
            plan.duplicate_checkpoint_triggers_suppressed
        ),
        total_full_state_validations=1 + embedded_validations + explicit_validations,
        reset_callback_count=0,
        boundary_callback_count=0,
        output_write_calls=0,
        artifact_bytes_written=0,
        persistent_state_and_transcript_equivalence_pending=True,
        scientific_promotion_allowed=False,
    )
    return _CollectedOperationalLife(
        trace=trace,
        metadata=metadata,
        state=executor.state,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadOperationalLifeResult:
    """One complete operational trace, metadata record, and final state."""

    config: HCCLContinualDyadOperationalLifeRunnerConfig
    trace: HCCLContinualDyadLifeTrace
    state: HCCLContinualDyadState
    metadata: HCCLContinualDyadOperationalLifeMetadata

    def __post_init__(self) -> None:
        if type(self.config) is not HCCLContinualDyadOperationalLifeRunnerConfig:
            raise TypeError("result config must be exact operational-life config")
        if type(self.trace) is not HCCLContinualDyadLifeTrace:
            raise TypeError("result trace must be exact HCCLContinualDyadLifeTrace")
        if type(self.state) is not HCCLContinualDyadState:
            raise TypeError("result state must be exact HCCLContinualDyadState")
        if type(self.metadata) is not HCCLContinualDyadOperationalLifeMetadata:
            raise TypeError("result metadata must be exact operational-life metadata")
        validate_hccl_continual_dyad_life_trace(self.trace)
        if not (
            self.trace.schedule_profile
            == self.metadata.schedule_profile
            == self.config.schedule_profile
        ):
            raise ValueError("operational result profile identities differ")
        expected = np.asarray((0, self.config.total_steps), dtype=np.uint32)
        if not np.array_equal(_state_clock(self.state, name="final state"), expected):
            raise ValueError("operational result state lacks the exact final clock")


class HCCLContinualDyadOperationalLifeRunner:
    """Initialize and execute one complete private compact operational life."""

    def __init__(
        self,
        config: HCCLContinualDyadOperationalLifeRunnerConfig | None = None,
    ) -> None:
        selected = (
            HCCLContinualDyadOperationalLifeRunnerConfig()
            if config is None
            else config
        )
        if type(selected) is not HCCLContinualDyadOperationalLifeRunnerConfig:
            raise TypeError("config must be exact operational-life config")
        self._config = selected
        self._factory = HCCLContinualDyadFactory(selected.factory_config)

    @property
    def config(self) -> HCCLContinualDyadOperationalLifeRunnerConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    def run(self, key: Array) -> HCCLContinualDyadOperationalLifeResult:
        """Execute one fixed complete life and return nothing partial on failure."""

        try:
            initialized = self._factory.init(key)
        except Exception as error:
            raise HCCLContinualDyadOperationalLifeError(
                -1,
                "factory-initialization",
                str(error),
            ) from error
        if type(initialized) is not HCCLContinualDyadFactoryInitialization:
            raise HCCLContinualDyadOperationalLifeError(
                -1,
                "factory-initialization",
                "factory returned a malformed initialization",
            )
        try:
            executor = _HCCLContinualDyadOperationalExecutor(
                initialized.transaction,
                initialized.state,
                checkpoint_interval=_FEATURE_LIFECYCLE_CHECKPOINT_CADENCE,
            )
        except Exception as error:
            raise HCCLContinualDyadOperationalLifeError(
                -1,
                "initial-executor-validation",
                str(error),
            ) from error
        collected = _collect_operational_life(
            self._config,
            executor,
            initialized.transaction.hccl.world.task_score_for_regime,
        )
        if type(collected.state) is not HCCLContinualDyadState:
            raise HCCLContinualDyadOperationalLifeError(
                self._config.total_steps,
                "final-state",
                "operational executor returned a malformed final state",
            )
        return HCCLContinualDyadOperationalLifeResult(
            config=self._config,
            trace=collected.trace,
            state=collected.state,
            metadata=collected.metadata,
        )


__all__ = (
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_CONFIG_SCHEMA",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_EVIDENCE_LEVEL",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_LIMITATIONS",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_METADATA_SCHEMA",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_LIFE_STATUS",
    "HCCLContinualDyadOperationalLifeError",
    "HCCLContinualDyadOperationalLifeMetadata",
    "HCCLContinualDyadOperationalLifeResult",
    "HCCLContinualDyadOperationalLifeRunner",
    "HCCLContinualDyadOperationalLifeRunnerConfig",
)
