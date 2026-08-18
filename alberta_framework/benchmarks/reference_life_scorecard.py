"""Matched, permanently nonpromoting reference-life development scorecard.

The scorecard is a development-selection instrument, never scientific evidence
and never a route to ``reference-dev``.  Its plan is intentionally fixed.  A
different arm, seed, horizon, environment, or gate is a new schema, not a CLI
override.

Execution streams each accepted event into bounded summaries.  Full event
objects are never retained by this module.  Wall-clock measurements are kept in
a separately labelled telemetry block and are excluded from every score and
gate.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import errno
import functools
import hashlib
import json
import math
import os
import stat
import struct
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import jax
import numpy as np

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.prototype_reference_adapter import PrototypeReferenceState
from alberta_framework.reference_agent import (
    REFERENCE_AGENT_API_VERSION,
    AgentCapabilities,
    AgentManifest,
    ArrayValue,
    ReferenceAgentUpdate,
    SpaceSpec,
    canonical_config_sha256,
)
from alberta_framework.reference_life import (
    REFERENCE_LIFE_RNG_SCHEDULE,
    ExactDispatchConfig,
    LifePhase,
    ReferenceEnvironmentManifest,
    ReferenceLifeMetricsConfig,
    ReferenceLifeRunner,
    RiverSwimReferenceEnvironment,
    SwitchingTwoStateReferenceEnvironment,
    build_prototype_riverswim_life,
    build_prototype_switching_life,
)
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

# Importing ReferenceAgentUpdate above is deliberate: the runner/control seam is
# the state-agnostic update contract, not PrototypeAdapterUpdate.
_REFERENCE_UPDATE_TYPE = ReferenceAgentUpdate


def _checkpoint_source_identity() -> dict[str, Any]:
    from alberta_framework.reference_life_checkpoint import _source_identity

    return _source_identity()


def _checkpoint_runtime_identity() -> dict[str, Any]:
    from alberta_framework.reference_life_checkpoint import _runtime_identity

    return _runtime_identity()


def _checkpoint_dependency_identity() -> dict[str, Any]:
    from alberta_framework.reference_life_checkpoint import _dependency_identity

    return _dependency_identity()

REFERENCE_LIFE_SCORECARD_PLAN_SCHEMA = "asi.reference_life_scorecard.plan.v1"
REFERENCE_LIFE_SCORECARD_RUN_SCHEMA = "asi.reference_life_scorecard.run.v1"
REFERENCE_LIFE_SCORECARD_ARTIFACT_SCHEMA = "asi.reference_life_scorecard.artifact.v1"
REFERENCE_LIFE_SCORECARD_SUMMARY_SCHEMA = "asi.reference_life_scorecard.summary.v1"
PROTOCOL_ID = "asi.reference_life_scorecard.matched_development.v1"

ARM_ROSTER = (
    "prototype",
    "prototype_frozen",
    "random",
    "privileged_oracle",
    "differential_sarsa",
    "sarsa",
)
ENVIRONMENT_ROSTER = ("switching_two_state", "riverswim")
SEED_ROSTER = tuple(range(70_000, 70_012))

SWITCHING_HORIZON = 4_000
SWITCHING_PHASE_LENGTH = 250
SWITCHING_POST_SWITCH_WINDOW = 50
RIVERSWIM_HORIZON = 20_000
RIVERSWIM_N_STATES = 6
RIVERSWIM_EARLY_WINDOW = 2_000
RIVERSWIM_LATE_WINDOW = 2_000

SWITCHING_PAYOFFS_A = ((0.0, 1.0), (1.0, 0.0))
SWITCHING_PAYOFFS_B = ((1.0, 0.0), (0.0, 1.0))
RIVERSWIM_P_RIGHT_UP = 0.35
RIVERSWIM_P_RIGHT_DOWN = 0.05
RIVERSWIM_REWARD_LEFT = 0.005
RIVERSWIM_REWARD_RIGHT = 1.0
RIVERSWIM_INITIAL_STATE = 0

# Plan v1 pins its explicit literals and every shard records the fully resolved
# live component configs plus source identity.  It does not claim that the plan
# digest alone freezes defaults which are not explicit in this payload.
REFERENCE_LIFE_SCORECARD_PLAN_V1_SHA256 = (
    "4c4396848da3abebc53fb61c7b241866566ad2b44fa83ee3395a5a389a869b86"
)

# A complete 144-shard aggregate is comfortably below this limit.  Keeping
# the cap explicit prevents validation inputs from becoming an unbounded read.
MAX_SCORECARD_JSON_INPUT_BYTES = 64 * 1024 * 1024
MAX_SCORECARD_AGGREGATE_INPUT_BYTES = 256 * 1024 * 1024

CONTROL_GATE_T_CRITICAL = 2.201
CONTROL_GATE_LCB_THRESHOLD = 0.10

NONPROMOTING_POLICY: dict[str, object] = {
    "evidence_class": "development_selection_scorecard",
    "development_only": True,
    "permanently_nonpromoting": True,
    "scientific_promotion_allowed": False,
    "reference_dev_population_allowed": False,
}

TELEMETRY_POLICY = {
    "classification": "telemetry_only",
    "used_for_selection": False,
    "used_for_scoring": False,
    "cross_machine_comparison_supported": False,
}

_SHA256_PATTERN_LENGTH = 64


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _validate_json_value(value: Any, *, path: str = "$", depth: int = 0) -> None:
    if depth > 64:
        _fail(f"{path}: JSON nesting exceeds 64 levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{path}: JSON numbers must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{path}: JSON object keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}", depth=depth + 1)
        return
    _fail(f"{path}: value is not canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite JSON with the scorecard's hash canonicalization."""

    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value does not have a canonical JSON encoding") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_exact_equal(left: Any, right: Any) -> bool:
    """Compare canonical JSON values without Python's bool/int coercions."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _json_exact_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        return struct.pack(">d", left) == struct.pack(">d", right)
    return bool(left == right)


def _digest_excluding(payload: Mapping[str, Any], field: str) -> str:
    return _sha256_json({key: value for key, value in payload.items() if key != field})


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_PATTERN_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_scorecard_lifecycle_id(value: object) -> bool:
    if type(value) is not str or not value.startswith("prototype."):
        return False
    digest = value[len("prototype.") :]
    return len(digest) == 16 and all(character in "0123456789abcdef" for character in digest)


def _arm_definitions() -> list[dict[str, Any]]:
    return [
        {
            "arm": "prototype",
            "family": "prototype",
            "role": "candidate",
            "learning_expectation": "at_least_one_parameter_change",
            "config": {
                "base_step_size": 0.05,
                "base_average_reward_step_size": 0.01,
                "base_trace_decay": 0.0,
                "epsilon_base": 0.1,
                "primitive_only": True,
            },
        },
        {
            "arm": "prototype_frozen",
            "family": "prototype",
            "role": "frozen_no_learning_control",
            "learning_expectation": "zero_parameter_changes",
            "config": {
                "base_step_size": 0.0,
                "base_average_reward_step_size": 0.0,
                "base_trace_decay": 0.0,
                "epsilon_base": 0.1,
                "primitive_only": True,
            },
        },
        {
            "arm": "random",
            "family": "uniform_random",
            "role": "random_control",
            "learning_expectation": "zero_parameter_changes",
            "config": {"action_distribution": "uniform", "n_actions": 2},
        },
        {
            "arm": "privileged_oracle",
            "family": "analytic_oracle",
            "role": "privileged_upper_control",
            "learning_expectation": "zero_parameter_changes",
            "config": {
                "finite_horizon_solver": (
                    "finite_horizon_backward_dynamic_program.float64.tie_low.preview1"
                ),
                "horizon_binding": "environment_protocol_horizon",
                "privileged_environment_model": True,
                "tie_break": "lowest_action_index",
            },
        },
        {
            "arm": "differential_sarsa",
            "family": "differential_sarsa",
            "role": "strong_control_candidate",
            "learning_expectation": "at_least_one_parameter_change",
            "config": {
                "q_step_size": 0.1,
                "average_reward_step_size": 0.01,
                "trace_decay": 0.0,
                "epsilon_start": 0.5,
                "epsilon_end": 0.02,
                "epsilon_decay_steps": 2_500,
                "use_bias": True,
            },
        },
        {
            "arm": "sarsa",
            "family": "discounted_sarsa",
            "role": "strong_control_candidate",
            "learning_expectation": "at_least_one_parameter_change",
            "config": {
                "gamma": 0.9,
                "epsilon_start": 0.5,
                "epsilon_end": 0.02,
                "epsilon_decay_steps": 2_500,
                "hidden_sizes": [],
                "step_size": 0.05,
                "sparsity": 0.0,
                "leaky_relu_slope": 0.01,
                "use_layer_norm": False,
                "lamda": 0.0,
            },
        },
    ]


def _rotated_arms(seed_index: int) -> tuple[str, ...]:
    offset = seed_index % len(ARM_ROSTER)
    return ARM_ROSTER[offset:] + ARM_ROSTER[:offset]


def _fixed_plan_payload_without_digest() -> dict[str, Any]:
    return {
        "schema": REFERENCE_LIFE_SCORECARD_PLAN_SCHEMA,
        "schema_version": 1,
        "benchmark": "reference_life_matched_development_scorecard",
        "protocol_id": PROTOCOL_ID,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "claim_limitations": [
            "not scientific evidence",
            "not performance evidence",
            "not reference-dev promotion",
            "not robotics readiness",
        ],
        "arm_roster": list(ARM_ROSTER),
        "arm_definitions": _arm_definitions(),
        "environment_roster": list(ENVIRONMENT_ROSTER),
        "seed_roster": list(SEED_ROSTER),
        "schedule": {
            "environment_seed_contract": (
                "each arm receives the identical uint32 life seed; environment keys are "
                "jax split index 1 and execution keys fold in the same cursor"
            ),
            "environment_order": list(ENVIRONMENT_ROSTER),
            "arm_order_policy": "left_cyclic_rotation_by_seed_roster_index",
            "execution_isolation": (
                "one_canonical_shard_per_fresh_python_process; aggregate only reads shards"
            ),
            "per_seed_arm_order": [
                {"seed": seed, "arms": list(_rotated_arms(index))}
                for index, seed in enumerate(SEED_ROSTER)
            ],
        },
        "protocols": {
            "switching_two_state": {
                "horizon": SWITCHING_HORIZON,
                "phase_length": SWITCHING_PHASE_LENGTH,
                "segments": 16,
                "post_switch_window": SWITCHING_POST_SWITCH_WINDOW,
                "aba_windows": {
                    "initial_a": [0, 50],
                    "first_b": [250, 300],
                    "return_a": [500, 550],
                    "interval_semantics": "zero_based_half_open_accepted_event_indices",
                },
                "payoffs_a": [list(row) for row in SWITCHING_PAYOFFS_A],
                "payoffs_b": [list(row) for row in SWITCHING_PAYOFFS_B],
                "initial_state": "uniform_random_from_environment_root_key",
                "continuing": True,
            },
            "riverswim": {
                "horizon": RIVERSWIM_HORIZON,
                "n_states": RIVERSWIM_N_STATES,
                "p_right_up": RIVERSWIM_P_RIGHT_UP,
                "p_right_down": RIVERSWIM_P_RIGHT_DOWN,
                "reward_left": RIVERSWIM_REWARD_LEFT,
                "reward_right": RIVERSWIM_REWARD_RIGHT,
                "initial_state": RIVERSWIM_INITIAL_STATE,
                "early_window": RIVERSWIM_EARLY_WINDOW,
                "late_window": RIVERSWIM_LATE_WINDOW,
                "high_end_visit_definition": (
                    "post_transition_state_index_equals_n_states_minus_one"
                ),
                "continuing": True,
            },
        },
        "summaries": {
            "event_retention": "streaming_o1_no_full_event_list",
            "normalization": (
                "within_environment_seed_paired_(arm_return-random_return)/"
                "mean(oracle_return-random_return)"
            ),
            "cross_environment_pooling": "forbidden",
        },
        "control_calibration_gate": {
            "classification": "development_heuristic_not_scientific_inference",
            "required_complete_seed_count": 12,
            "scale": "mean_seed(oracle_return-random_return)",
            "paired_sample": "(sarsa_return_i-random_return_i)/scale",
            "lcb": "mean(sample)-2.201*sample_sd_ddof1/sqrt(12)",
            "t_critical": CONTROL_GATE_T_CRITICAL,
            "minimum_lcb_exclusive": CONTROL_GATE_LCB_THRESHOLD,
            "environment_qualifies": (
                "differential_sarsa_or_sarsa_has_lcb_strictly_above_threshold"
            ),
            "overall_failure_status": "valid_baseline_failure",
        },
        "candidate_development_gate": {
            "classification": "development_heuristic_not_scientific_inference",
            "prototype_vs_frozen_paired_lcb_minimum_inclusive": 0.05,
            "prototype_vs_best_qualifying_sarsa_noninferiority_margin": -0.05,
            "required_in_each_environment": True,
            "pareto_utility_advantage": 0.05,
            "pareto_resource_advantage_fraction": 0.20,
            "resource_decision": "not_evaluated_while_latency_is_telemetry_only",
        },
        "timing_policy": dict(TELEMETRY_POLICY),
        "resource_policy": {
            "persistent_state": "numeric_jax_and_numpy_pytree_leaf_bytes",
            "static_adapter_numeric_state": (
                "finite_horizon_oracle_policy_bytes_else_zero"
            ),
            "trainable_scalars": (
                "adapter_parameter_tree_when_available_else_floating_leaf_upper_bound"
            ),
            "estimates_are_not_hardware_memory_peaks": True,
        },
    }


def _fixed_plan_payload() -> dict[str, Any]:
    payload = _fixed_plan_payload_without_digest()
    observed_digest = _sha256_json(payload)
    if observed_digest != REFERENCE_LIFE_SCORECARD_PLAN_V1_SHA256:
        raise RuntimeError(
            "reference-life scorecard plan v1 literals no longer match the pinned digest"
        )
    payload["plan_sha256"] = REFERENCE_LIFE_SCORECARD_PLAN_V1_SHA256
    return payload


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeDevelopmentPlan:
    """Deeply immutable wrapper around the one accepted development plan."""

    schema: str
    plan_sha256: str
    _canonical_json: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_LIFE_SCORECARD_PLAN_SCHEMA:
            raise ValueError("unsupported reference-life development-plan schema")
        if not _is_sha256(self.plan_sha256):
            raise ValueError("plan_sha256 must be a lowercase SHA-256 digest")
        try:
            payload = json.loads(self._canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("development plan is not canonical JSON") from exc
        if canonical_json_bytes(payload).decode("utf-8") != self._canonical_json:
            raise ValueError("development plan JSON is not canonical")
        if not _json_exact_equal(payload, _fixed_plan_payload()):
            raise ValueError("payload is not the canonical fixed development plan")
        if payload["plan_sha256"] != self.plan_sha256:
            raise ValueError("development plan digest is inconsistent")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReferenceLifeDevelopmentPlan:
        try:
            copied = json.loads(canonical_json_bytes(payload).decode("utf-8"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("payload is not the canonical fixed development plan") from exc
        if not _json_exact_equal(copied, _fixed_plan_payload()):
            raise ValueError("payload is not the canonical fixed development plan")
        return cls(
            schema=REFERENCE_LIFE_SCORECARD_PLAN_SCHEMA,
            plan_sha256=cast(str, copied["plan_sha256"]),
            _canonical_json=canonical_json_bytes(copied).decode("utf-8"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = json.loads(self._canonical_json)
        assert isinstance(payload, dict)
        return payload

    @property
    def seeds(self) -> tuple[int, ...]:
        return SEED_ROSTER

    @property
    def arms(self) -> tuple[str, ...]:
        return ARM_ROSTER

    @property
    def environments(self) -> tuple[str, ...]:
        return ENVIRONMENT_ROSTER

    def arm_order(self, seed: int) -> tuple[str, ...]:
        if type(seed) is not int or seed not in SEED_ROSTER:
            raise ValueError("seed is not in the fixed scorecard roster")
        return _rotated_arms(SEED_ROSTER.index(seed))

    def arm_definition(self, arm: str) -> dict[str, Any]:
        if arm not in ARM_ROSTER:
            raise ValueError(f"unsupported scorecard arm {arm!r}")
        definitions = self.to_payload()["arm_definitions"]
        assert isinstance(definitions, list)
        for definition in definitions:
            if definition["arm"] == arm:
                return cast(dict[str, Any], definition)
        raise AssertionError("canonical arm definition is missing")

    def protocol(self, environment_kind: str) -> dict[str, Any]:
        if environment_kind not in ENVIRONMENT_ROSTER:
            raise ValueError(f"unsupported environment {environment_kind!r}")
        protocol = self.to_payload()["protocols"][environment_kind]
        assert isinstance(protocol, dict)
        return protocol


def build_development_plan() -> ReferenceLifeDevelopmentPlan:
    """Return the only scorecard plan accepted by this schema."""

    return ReferenceLifeDevelopmentPlan.from_payload(_fixed_plan_payload())


@dataclasses.dataclass(slots=True)
class _Window:
    start: int
    stop: int
    event_count: int = 0
    reward_sum: float = 0.0
    regret_sum: float = 0.0

    def observe(self, index: int, reward: float, oracle_reward: float) -> None:
        if self.start <= index < self.stop:
            self.event_count += 1
            self.reward_sum += reward
            self.regret_sum += oracle_reward - reward

    def payload(self) -> dict[str, int | float]:
        if self.event_count <= 0:
            raise ValueError("metric window did not receive its fixed event count")
        return {
            "event_count": self.event_count,
            "reward_sum": self.reward_sum,
            "mean_reward": self.reward_sum / self.event_count,
            "mean_oracle_regret": self.regret_sum / self.event_count,
        }


class StreamingRunSummary:
    """O(1)-space online reducer for one fixed-horizon life.

    The object owns only scalar counters and at most three fixed windows.  It
    intentionally has no event collection and cannot reconstruct a transcript.
    The authoritative runner's transcript digest is recorded separately.
    """

    __slots__ = (
        "_accepted_events",
        "_early",
        "_environment_kind",
        "_high_end_visit_count",
        "_horizon",
        "_late",
        "_n_states",
        "_oracle_reward_sum",
        "_parameter_change_events",
        "_phase_event_counts",
        "_phase_length",
        "_phase_reward_sums",
        "_reward_sum",
        "_switching_windows",
    )

    def __init__(
        self,
        *,
        environment_kind: str,
        horizon: int,
        phase_length: int | None,
        n_states: int | None,
        early_window: int | None,
        late_window: int | None,
        post_switch_window: int | None,
    ) -> None:
        if environment_kind not in ENVIRONMENT_ROSTER:
            raise ValueError("unsupported streaming-summary environment")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self._environment_kind = environment_kind
        self._horizon = horizon
        self._accepted_events = 0
        self._reward_sum = 0.0
        self._oracle_reward_sum = 0.0
        self._parameter_change_events = 0
        self._phase_event_counts = [0, 0]
        self._phase_reward_sums = [0.0, 0.0]
        self._high_end_visit_count = 0
        self._phase_length = phase_length
        self._n_states = n_states
        self._switching_windows: dict[str, _Window] = {}
        self._early: _Window | None = None
        self._late: _Window | None = None

        if environment_kind == "switching_two_state":
            if type(phase_length) is not int or phase_length <= 0:
                raise ValueError("switching phase_length must be positive")
            if type(post_switch_window) is not int or post_switch_window <= 0:
                raise ValueError("switching post_switch_window must be positive")
            if 2 * phase_length + post_switch_window > horizon:
                raise ValueError("switching horizon does not contain the A/B/A windows")
            self._switching_windows = {
                "initial_a": _Window(0, post_switch_window),
                "first_b": _Window(phase_length, phase_length + post_switch_window),
                "return_a": _Window(
                    2 * phase_length,
                    2 * phase_length + post_switch_window,
                ),
            }
        else:
            if type(n_states) is not int or n_states < 2:
                raise ValueError("RiverSwim n_states must be at least two")
            if type(early_window) is not int or not 0 < early_window <= horizon:
                raise ValueError("RiverSwim early window is outside the horizon")
            if type(late_window) is not int or not 0 < late_window <= horizon:
                raise ValueError("RiverSwim late window is outside the horizon")
            self._early = _Window(0, early_window)
            self._late = _Window(horizon - late_window, horizon)

    @classmethod
    def for_switching(
        cls,
        *,
        horizon: int,
        phase_length: int,
        post_switch_window: int,
    ) -> StreamingRunSummary:
        return cls(
            environment_kind="switching_two_state",
            horizon=horizon,
            phase_length=phase_length,
            n_states=None,
            early_window=None,
            late_window=None,
            post_switch_window=post_switch_window,
        )

    @classmethod
    def for_riverswim(
        cls,
        *,
        horizon: int,
        early_window: int,
        late_window: int,
        n_states: int,
    ) -> StreamingRunSummary:
        return cls(
            environment_kind="riverswim",
            horizon=horizon,
            phase_length=None,
            n_states=n_states,
            early_window=early_window,
            late_window=late_window,
            post_switch_window=None,
        )

    @property
    def accepted_events(self) -> int:
        return self._accepted_events

    def observe(
        self,
        *,
        reward: float,
        oracle_reward: float,
        regime_id: int,
        parameters_changed: bool,
        next_state_index: int,
    ) -> None:
        if self._accepted_events >= self._horizon:
            raise ValueError("streaming summary already reached its horizon")
        if type(reward) not in (int, float):
            raise ValueError("reward must be a finite number")
        if type(oracle_reward) not in (int, float):
            raise ValueError("oracle_reward must be a finite number")
        reward_value = float(reward)
        oracle_value = float(oracle_reward)
        if not math.isfinite(reward_value) or not math.isfinite(oracle_value):
            raise ValueError("streaming metric inputs must be finite")
        if not isinstance(parameters_changed, bool):
            raise ValueError("parameters_changed must be boolean")
        if type(next_state_index) is not int:
            raise ValueError("next_state_index must be an integer")

        index = self._accepted_events
        if self._environment_kind == "switching_two_state":
            assert self._phase_length is not None
            expected_regime = (index // self._phase_length) % 2
            if regime_id != expected_regime:
                raise ValueError("switching regime does not match the fixed schedule")
            if next_state_index not in (0, 1):
                raise ValueError("switching next state is outside {0, 1}")
            for window in self._switching_windows.values():
                window.observe(index, reward_value, oracle_value)
        else:
            if regime_id != 0:
                raise ValueError("RiverSwim must remain in stationary regime zero")
            assert self._n_states is not None
            if next_state_index < 0 or next_state_index >= self._n_states:
                raise ValueError("RiverSwim next state is outside its chain")
            assert self._early is not None and self._late is not None
            self._early.observe(index, reward_value, oracle_value)
            self._late.observe(index, reward_value, oracle_value)
            self._high_end_visit_count += int(next_state_index == self._n_states - 1)

        self._accepted_events += 1
        self._reward_sum += reward_value
        self._oracle_reward_sum += oracle_value
        self._parameter_change_events += int(parameters_changed)
        self._phase_event_counts[regime_id] += 1
        self._phase_reward_sums[regime_id] += reward_value

    def _payload(self, *, require_complete: bool) -> dict[str, Any]:
        if require_complete and self._accepted_events != self._horizon:
            raise ValueError(
                "cannot finalize streaming summary before the configured horizon"
            )
        payload: dict[str, Any] = {
            "summary_mode": "streaming_o1_no_retained_events",
            "configured_horizon": self._horizon,
            "accepted_events": self._accepted_events,
            "reward_sum": self._reward_sum,
            "mean_reward": (
                None
                if self._accepted_events == 0
                else self._reward_sum / self._accepted_events
            ),
            "oracle_reward_sum": self._oracle_reward_sum,
            "regret_sum": self._oracle_reward_sum - self._reward_sum,
            "parameter_change_events": self._parameter_change_events,
            "phase_event_counts": list(self._phase_event_counts),
            "phase_reward_sums": list(self._phase_reward_sums),
        }
        if self._environment_kind == "switching_two_state":
            windows: dict[str, Any]
            if require_complete:
                windows = {
                    name: window.payload()
                    for name, window in self._switching_windows.items()
                }
            else:
                windows = {
                    name: (
                        None if window.event_count == 0 else window.payload()
                    )
                    for name, window in self._switching_windows.items()
                }
            payload["windows"] = windows
            payload["high_end_visit_count"] = None
            payload["high_end_visit_rate"] = None
        else:
            assert self._early is not None and self._late is not None
            payload["windows"] = {
                "early": (
                    self._early.payload()
                    if self._early.event_count > 0
                    else None
                ),
                "late": (
                    self._late.payload()
                    if self._late.event_count > 0
                    else None
                ),
            }
            payload["high_end_visit_count"] = self._high_end_visit_count
            payload["high_end_visit_rate"] = (
                None
                if self._accepted_events == 0
                else self._high_end_visit_count / self._accepted_events
            )
        return payload

    def finalize(self) -> dict[str, Any]:
        return self._payload(require_complete=True)

    def partial(self) -> dict[str, Any]:
        """Return bounded diagnostics for a failed incomplete life."""

        return self._payload(require_complete=False)


def _iter_array_leaves(value: Any) -> Any:
    if isinstance(value, ArrayValue):
        # ArrayValue is a dataclass whose numeric storage is intentionally an
        # immutable bytes payload.  Treat that payload as the numeric leaf it
        # represents instead of losing cached observations/actions while
        # recursively walking only Python containers.
        yield value.to_numpy()
        return
    if isinstance(value, (jax.Array, np.ndarray)):
        yield value
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _iter_array_leaves(getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_array_leaves(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_array_leaves(item)


def _array_leaf_facts(tree: Any) -> tuple[int, int, int, int]:
    array_leaves = 0
    elements = 0
    byte_count = 0
    floating_elements = 0
    for leaf in _iter_array_leaves(tree):
        array_leaves += 1
        leaf_size = int(getattr(leaf, "size", 0))
        elements += leaf_size
        byte_count += int(getattr(leaf, "nbytes", 0))
        try:
            dtype = np.dtype(leaf.dtype)
        except TypeError:
            # Typed JAX PRNG keys are persistent arrays but are not trainable.
            continue
        if dtype.kind in {"f", "c"}:
            floating_elements += leaf_size
    return array_leaves, elements, byte_count, floating_elements


def estimate_jax_resources(state: Any) -> dict[str, Any]:
    """Return a transparent generic persistent-state/resource estimate.

    The fallback trainable count is explicitly an upper bound: floating
    optimizer statistics, traces, and caches may be included.  Execution uses
    narrower adapter-specific parameter trees where this module knows them.
    """

    leaves, elements, byte_count, floating_elements = _array_leaf_facts(state)
    return {
        "persistent_jax_array_leaves": leaves,
        "persistent_jax_array_scalar_count": elements,
        "persistent_jax_array_bytes": byte_count,
        "persistent_state_measurement": "numeric_jax_and_numpy_pytree_leaves",
        "trainable_scalar_count_estimate": floating_elements,
        "trainable_scalar_count_method": "floating_jax_pytree_leaves_upper_bound",
        "hardware_peak_memory_measured": False,
    }


def _prototype_parameter_tree(state: Any) -> Any | None:
    if not isinstance(state, PrototypeReferenceState):
        return None
    oak_state = state.agent_state.oak_state
    stomp_state = getattr(oak_state, "stomp_state", None)
    base = getattr(stomp_state, "base_learner_state", None)
    if base is None:
        return None
    return (
        getattr(base, "trunk_params", ()),
        getattr(base, "head_params", ()),
        getattr(stomp_state, "base_average_reward", ()),
    )


def _known_control_parameter_tree(state: Any) -> Any | None:
    """Extract known control parameters without treating traces as trainable."""

    inner = getattr(state, "agent_state", state)
    if all(hasattr(inner, name) for name in ("q_weights", "average_reward")):
        return (
            inner.q_weights,
            getattr(inner, "q_bias", ()),
            inner.average_reward,
        )
    learner_state = getattr(inner, "learner_state", None)
    if learner_state is not None and all(
        hasattr(learner_state, name) for name in ("trunk_params", "head_params")
    ):
        return learner_state.trunk_params, learner_state.head_params
    return None


def _agent_resource_payload(adapter: Any, state: Any) -> dict[str, Any]:
    resource = estimate_jax_resources(state)
    static_bytes = getattr(adapter, "static_numeric_bytes", 0)
    if type(static_bytes) is not int or static_bytes < 0:
        raise ValueError("adapter static_numeric_bytes must be a nonnegative integer")
    resource["persistent_static_numeric_bytes"] = static_bytes
    resource["persistent_numeric_bytes_total"] = (
        resource["persistent_jax_array_bytes"] + static_bytes
    )
    parameter_tree: Any | None = None
    method = "floating_jax_pytree_leaves_upper_bound"
    adapter_tree = getattr(adapter, "trainable_parameter_tree", None)
    if callable(adapter_tree):
        parameter_tree = adapter_tree(state)
        method = "adapter_declared_parameter_tree"
    if parameter_tree is None:
        parameter_tree = _prototype_parameter_tree(state)
        if parameter_tree is not None:
            method = "prototype_control_parameter_tree"
    if parameter_tree is None:
        parameter_tree = _known_control_parameter_tree(state)
        if parameter_tree is not None:
            method = "known_sarsa_parameter_fields"
    if parameter_tree is not None:
        _, parameter_elements, _, floating_elements = _array_leaf_facts(parameter_tree)
        resource["trainable_scalar_count_estimate"] = floating_elements
        resource["trainable_parameter_tree_scalar_count"] = parameter_elements
        resource["trainable_scalar_count_method"] = method
    return resource


@functools.lru_cache(maxsize=len(ENVIRONMENT_ROSTER) * len(ARM_ROSTER))
def _canonical_initial_resource_payload(
    environment_kind: str,
    arm: str,
) -> dict[str, Any]:
    """Measure the fixed-shape canonical initial state once per arm/environment."""

    plan = build_development_plan()
    spec = next(
        item
        for item in iter_run_specs(plan)
        if item.environment_kind == environment_kind
        and item.arm == arm
        and item.seed == SEED_ROSTER[0]
    )
    runner = build_scorecard_runner(plan, spec)
    state = runner.init()
    return _agent_resource_payload(runner.agent_adapter, state.agent_state)


def parameter_change_check(arm: str, parameter_change_events: int) -> dict[str, Any]:
    """Apply the fixed candidate/frozen/control learning-side-effect check."""

    if arm not in ARM_ROSTER:
        raise ValueError(f"unsupported scorecard arm {arm!r}")
    if type(parameter_change_events) is not int or parameter_change_events < 0:
        raise ValueError("parameter_change_events must be nonnegative")
    must_change = arm in ("prototype", "differential_sarsa", "sarsa")
    passed = parameter_change_events > 0 if must_change else parameter_change_events == 0
    return {
        "expectation": (
            "at_least_one_parameter_change" if must_change else "zero_parameter_changes"
        ),
        "observed_parameter_change_events": parameter_change_events,
        "passed": passed,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(values) / len(values)


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    return math.sqrt(math.fsum((value - average) ** 2 for value in values) / (len(values) - 1))


def _stderr(values: Sequence[float]) -> float:
    return _sample_sd(values) / math.sqrt(len(values)) if values else 0.0


def _paired_lcb(values: Sequence[float]) -> float | None:
    if len(values) != len(SEED_ROSTER):
        return None
    return _mean(values) - CONTROL_GATE_T_CRITICAL * _sample_sd(values) / math.sqrt(
        len(values)
    )


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str, int]:
    environment = record.get("environment_kind")
    arm = record.get("arm")
    seed = record.get("seed")
    if environment not in ENVIRONMENT_ROSTER:
        raise ValueError("run record has an unsupported environment")
    if arm not in ARM_ROSTER:
        raise ValueError("run record has an unsupported arm")
    if type(seed) is not int or seed not in SEED_ROSTER:
        raise ValueError("run record has a seed outside the fixed roster")
    return cast(str, environment), cast(str, arm), seed


def _reward_sum(record: Mapping[str, Any]) -> float:
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        raise ValueError("completed run record lacks an outcome")
    value = outcome.get("reward_sum")
    if type(value) not in (int, float):
        raise ValueError("completed run reward_sum must be finite")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError("completed run reward_sum must be finite")
    return result


def _summarize_validated_run_records(
    plan: ReferenceLifeDevelopmentPlan,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce a complete set after every record passed strict validation."""

    if not isinstance(plan, ReferenceLifeDevelopmentPlan):
        raise TypeError("plan must be a ReferenceLifeDevelopmentPlan")
    expected = {
        (environment, arm, seed)
        for environment in ENVIRONMENT_ROSTER
        for arm in ARM_ROSTER
        for seed in SEED_ROSTER
    }
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("every run record must be an object")
        identity = _record_identity(record)
        if identity in indexed:
            raise ValueError(f"duplicate run identity {identity!r}")
        status = record.get("status")
        if status not in ("completed", "failed"):
            raise ValueError("run status must be completed or failed")
        if status == "completed":
            _reward_sum(record)
        indexed[identity] = record
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"run records do not cover the fixed matched schedule; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )

    failures = []
    parameter_failures = []
    for identity in sorted(
        indexed,
        key=lambda item: (
            ENVIRONMENT_ROSTER.index(item[0]),
            ARM_ROSTER.index(item[1]),
            SEED_ROSTER.index(item[2]),
        ),
    ):
        record = indexed[identity]
        if record["status"] == "failed":
            failures.append(
                {
                    "environment_kind": identity[0],
                    "arm": identity[1],
                    "seed": identity[2],
                    "failure": record.get("failure"),
                }
            )
        outcome = record.get("outcome")
        if isinstance(outcome, Mapping):
            check = outcome.get("parameter_change_check")
            if isinstance(check, Mapping) and check.get("passed") is not True:
                parameter_failures.append(
                    {
                        "environment_kind": identity[0],
                        "arm": identity[1],
                        "seed": identity[2],
                        "parameter_change_check": dict(check),
                    }
                )
    execution_or_learning_failed = bool(failures or parameter_failures)

    environment_summaries: dict[str, Any] = {}
    environments_qualified = not execution_or_learning_failed
    candidate_utility_qualified = not execution_or_learning_failed
    for environment in ENVIRONMENT_ROSTER:
        returns: dict[str, dict[int, float]] = {}
        for arm in ARM_ROSTER:
            returns[arm] = {
                seed: _reward_sum(indexed[(environment, arm, seed)])
                for seed in SEED_ROSTER
                if indexed[(environment, arm, seed)]["status"] == "completed"
            }

        paired_baseline_seeds = tuple(
            seed
            for seed in SEED_ROSTER
            if seed in returns["random"] and seed in returns["privileged_oracle"]
        )
        baseline_differences = [
            returns["privileged_oracle"][seed] - returns["random"][seed]
            for seed in paired_baseline_seeds
        ]
        scale = (
            _mean(baseline_differences)
            if len(paired_baseline_seeds) == len(SEED_ROSTER)
            else None
        )
        scale_valid = scale is not None and math.isfinite(scale) and scale > 0.0

        arm_summaries: dict[str, Any] = {}
        for arm in ARM_ROSTER:
            arm_values = [returns[arm][seed] for seed in SEED_ROSTER if seed in returns[arm]]
            paired_seeds = tuple(
                seed
                for seed in SEED_ROSTER
                if seed in returns[arm] and seed in returns["random"]
            )
            normalized: list[float] | None = None
            if scale_valid and len(paired_seeds) == len(SEED_ROSTER):
                assert scale is not None
                normalized = [
                    (returns[arm][seed] - returns["random"][seed]) / scale
                    for seed in paired_seeds
                ]
            if normalized is None:
                normalized_mean = None
                normalized_sd = None
                lcb = None
            else:
                normalized_mean = _mean(normalized)
                normalized_sd = _sample_sd(normalized)
                lcb = normalized_mean - (
                    CONTROL_GATE_T_CRITICAL
                    * normalized_sd
                    / math.sqrt(len(normalized))
                )
            arm_summaries[arm] = {
                "completed_seed_count": len(arm_values),
                "failed_seed_count": len(SEED_ROSTER) - len(arm_values),
                "reward_sum_mean": None if not arm_values else _mean(arm_values),
                "reward_sum_stderr": None if not arm_values else _stderr(arm_values),
                "paired_seed_count": len(paired_seeds),
                "normalized_score_mean": normalized_mean,
                "normalized_score_sample_sd": normalized_sd,
                "paired_t_lcb_95": lcb,
                "normalized_samples_by_seed": (
                    None
                    if normalized is None
                    else [
                        {"seed": seed, "score": score}
                        for seed, score in zip(paired_seeds, normalized, strict=True)
                    ]
                ),
            }

        statistically_qualified_arms = [
            arm
            for arm in ("differential_sarsa", "sarsa")
            if arm_summaries[arm]["paired_seed_count"] == len(SEED_ROSTER)
            and arm_summaries[arm]["paired_t_lcb_95"] is not None
            and arm_summaries[arm]["paired_t_lcb_95"] > CONTROL_GATE_LCB_THRESHOLD
        ]
        qualified_arms = (
            [] if execution_or_learning_failed else statistically_qualified_arms
        )
        environment_qualified = (
            not execution_or_learning_failed and scale_valid and bool(qualified_arms)
        )
        environments_qualified = environments_qualified and environment_qualified

        prototype_frozen_samples: list[float] | None = None
        if scale_valid and all(
            seed in returns["prototype"] and seed in returns["prototype_frozen"]
            for seed in SEED_ROSTER
        ):
            assert scale is not None
            prototype_frozen_samples = [
                (
                    returns["prototype"][seed]
                    - returns["prototype_frozen"][seed]
                )
                / scale
                for seed in SEED_ROSTER
            ]
        prototype_frozen_lcb = (
            None
            if prototype_frozen_samples is None
            else _paired_lcb(prototype_frozen_samples)
        )
        beats_frozen = not execution_or_learning_failed and (
            prototype_frozen_lcb is not None and prototype_frozen_lcb >= 0.05
        )

        best_sarsa: str | None = None
        if qualified_arms:
            best_sarsa = max(
                qualified_arms,
                key=lambda arm: (
                    cast(float, arm_summaries[arm]["normalized_score_mean"]),
                    -ARM_ROSTER.index(arm),
                ),
            )
        prototype_sarsa_samples: list[float] | None = None
        if (
            scale_valid
            and best_sarsa is not None
            and all(
                seed in returns["prototype"] and seed in returns[best_sarsa]
                for seed in SEED_ROSTER
            )
        ):
            assert scale is not None
            prototype_sarsa_samples = [
                (returns["prototype"][seed] - returns[best_sarsa][seed]) / scale
                for seed in SEED_ROSTER
            ]
        prototype_sarsa_lcb = (
            None
            if prototype_sarsa_samples is None
            else _paired_lcb(prototype_sarsa_samples)
        )
        sarsa_noninferior = not execution_or_learning_failed and (
            prototype_sarsa_lcb is not None and prototype_sarsa_lcb >= -0.05
        )
        environment_candidate_qualified = (
            environment_qualified and beats_frozen and sarsa_noninferior
        )
        candidate_utility_qualified = (
            candidate_utility_qualified and environment_candidate_qualified
        )
        environment_summaries[environment] = {
            "normalization": {
                "scope": "within_environment_only",
                "paired_baseline_seed_count": len(paired_baseline_seeds),
                "scale": scale,
                "scale_valid_finite_positive": scale_valid,
                "definition": (
                    "(arm_reward_sum-random_reward_sum)/"
                    "mean_seed(oracle_reward_sum-random_reward_sum)"
                ),
            },
            "arms": arm_summaries,
            "control_calibration": {
                "classification": "development_heuristic_not_scientific_inference",
                "qualified": environment_qualified,
                "qualified_sarsa_arms": qualified_arms,
                "minimum_lcb_exclusive": CONTROL_GATE_LCB_THRESHOLD,
                "t_critical": CONTROL_GATE_T_CRITICAL,
            },
            "candidate_development_checks": {
                "classification": "development_heuristic_not_scientific_inference",
                "prototype_vs_frozen_paired_lcb_95": prototype_frozen_lcb,
                "prototype_vs_frozen_minimum_inclusive": 0.05,
                "prototype_vs_frozen_passed": beats_frozen,
                "best_qualifying_sarsa_arm": best_sarsa,
                "prototype_vs_best_sarsa_paired_lcb_95": prototype_sarsa_lcb,
                "sarsa_noninferiority_margin_inclusive": -0.05,
                "sarsa_noninferiority_passed": sarsa_noninferior,
                "utility_gate_passed": environment_candidate_qualified,
            },
        }

    if failures:
        status = "valid_execution_failure"
    elif parameter_failures:
        status = "valid_parameter_change_failure"
    elif not environments_qualified:
        status = "valid_baseline_failure"
    else:
        status = "development_scorecard_complete"
    if failures:
        candidate_selection_status = "not_evaluated_execution_failure"
    elif parameter_failures:
        candidate_selection_status = "not_evaluated_parameter_change_failure"
    elif not environments_qualified:
        candidate_selection_status = "not_evaluated_baseline_failure"
    elif not candidate_utility_qualified:
        candidate_selection_status = "rejected_development_utility_gate"
    else:
        candidate_selection_status = "not_evaluated_resource_telemetry_only"
    return {
        "schema": REFERENCE_LIFE_SCORECARD_SUMMARY_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "status": status,
        "status_is_promotion": False,
        "run_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "parameter_change_failure_count": len(parameter_failures),
        "parameter_change_failures": parameter_failures,
        "environments": environment_summaries,
        "cross_environment_pooled_score": None,
        "cross_environment_pooling_forbidden": True,
        "control_calibration_gate_passed": environments_qualified,
        "candidate_utility_gate_passed": candidate_utility_qualified,
        "candidate_selection_status": candidate_selection_status,
        "pareto_resource_decision": {
            "status": "not_evaluated_resource_telemetry_only",
            "reason": (
                "cold/warmed latency is telemetry-only and cannot be used in selection"
            ),
            "future_gate_if_latency_is_qualified_under_a_new_protocol": {
                "requires_no_qualifying_sarsa_utility_noninferior_on_both_environments": True,
                "candidate_utility_advantage": 0.05,
                "candidate_resource_advantage_fraction": 0.20,
                "requires_no_more_persistent_bytes": True,
                "requires_no_more_p95_latency": True,
            },
        },
    }


def summarize_run_records(
    plan: ReferenceLifeDevelopmentPlan,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Strictly validate a complete matrix, then recompute its development gates."""

    if not isinstance(plan, ReferenceLifeDevelopmentPlan):
        raise TypeError("plan must be a ReferenceLifeDevelopmentPlan")
    specs = iter_run_specs(plan)
    if len(records) != len(specs):
        raise ValueError("run records must contain exactly the fixed 144-shard matrix")
    by_identity: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("every run record must be an object")
        identity = _record_identity(record)
        if identity in by_identity:
            raise ValueError(f"duplicate run identity {identity!r}")
        by_identity[identity] = record
    identities = _current_consistency_identities()
    ordered: list[Mapping[str, Any]] = []
    for spec in specs:
        identity = (spec.environment_kind, spec.arm, spec.seed)
        scheduled_record = by_identity.get(identity)
        if scheduled_record is None:
            raise ValueError(f"run records are missing scheduled identity {identity!r}")
        _validate_run_record(
            plan,
            scheduled_record,
            spec,
            path=f"$.runs[{spec.schedule_index}]",
            consistency_identities=identities,
        )
        ordered.append(scheduled_record)
    return _summarize_validated_run_records(plan, ordered)


@dataclasses.dataclass(frozen=True, slots=True)
class ScorecardRunSpec:
    schedule_index: int
    environment_kind: str
    arm: str
    seed: int
    lifecycle_id: str

    def __post_init__(self) -> None:
        schedule_size = len(SEED_ROSTER) * len(ENVIRONMENT_ROSTER) * len(ARM_ROSTER)
        if type(self.schedule_index) is not int or not 0 <= self.schedule_index < schedule_size:
            raise ValueError("schedule_index must be a nonnegative integer in the fixed schedule")
        if (
            type(self.environment_kind) is not str
            or self.environment_kind not in ENVIRONMENT_ROSTER
        ):
            raise ValueError("environment_kind is not in the fixed scorecard roster")
        if type(self.arm) is not str or self.arm not in ARM_ROSTER:
            raise ValueError("arm is not in the fixed scorecard roster")
        if type(self.seed) is not int or self.seed not in SEED_ROSTER:
            raise ValueError("seed is not in the fixed scorecard roster")
        if not _is_scorecard_lifecycle_id(self.lifecycle_id):
            raise ValueError("lifecycle_id must be a prototype.<16-hex> identity")
        runs_per_seed = len(ENVIRONMENT_ROSTER) * len(ARM_ROSTER)
        seed_index, within_seed = divmod(self.schedule_index, runs_per_seed)
        environment_index, arm_index = divmod(within_seed, len(ARM_ROSTER))
        expected_seed = SEED_ROSTER[seed_index]
        expected_environment = ENVIRONMENT_ROSTER[environment_index]
        expected_arm = _rotated_arms(seed_index)[arm_index]
        if (
            self.seed != expected_seed
            or self.environment_kind != expected_environment
            or self.arm != expected_arm
        ):
            raise ValueError("run identity does not match its fixed schedule_index")
        identity = (
            f"{REFERENCE_LIFE_SCORECARD_PLAN_V1_SHA256}:"
            f"{expected_environment}:{expected_arm}:{expected_seed}"
        )
        lifecycle_hex = hashlib.sha256(identity.encode("ascii")).hexdigest()[:16]
        expected_lifecycle = f"prototype.{lifecycle_hex}"
        if self.lifecycle_id != expected_lifecycle:
            raise ValueError("lifecycle_id does not match the fixed run identity")


def iter_run_specs(plan: ReferenceLifeDevelopmentPlan) -> tuple[ScorecardRunSpec, ...]:
    """Expand the fixed cyclic schedule into immutable run identities."""

    if not isinstance(plan, ReferenceLifeDevelopmentPlan):
        raise TypeError("plan must be a ReferenceLifeDevelopmentPlan")
    specs: list[ScorecardRunSpec] = []
    for seed in plan.seeds:
        for environment in plan.environments:
            for arm in plan.arm_order(seed):
                schedule_index = len(specs)
                identity = f"{plan.plan_sha256}:{environment}:{arm}:{seed}"
                lifecycle_hex = hashlib.sha256(identity.encode("ascii")).hexdigest()[:16]
                specs.append(
                    ScorecardRunSpec(
                        schedule_index=schedule_index,
                        environment_kind=environment,
                        arm=arm,
                        seed=seed,
                        lifecycle_id=f"prototype.{lifecycle_hex}",
                    )
                )
    return tuple(specs)


def preflight_new_output(path: Path) -> Path:
    """Reject an occupied output before any expensive scorecard execution."""

    destination, parent_fd = _open_output_parent(Path(path), create=True)
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return destination
        raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
    finally:
        os.close(parent_fd)


def _open_output_parent(path: Path, *, create: bool) -> tuple[Path, int]:
    """Open an absolute parent through no-follow directory descriptors."""

    destination = Path(os.path.abspath(os.fspath(path)))
    parent = destination.parent
    descriptor = os.open(
        os.path.sep,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        for component in parent.parts[1:]:
            if component in ("", ".", ".."):
                raise ValueError("output path contains an unsafe directory component")
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return destination, descriptor
    except Exception:
        os.close(descriptor)
        raise


def _link_unnamed_file(file_fd: int, parent_fd: int, name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    if linkat(file_fd, b"", parent_fd, os.fsencode(name), 0x1000) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), name)
        raise OSError(error, os.strerror(error), name)


def write_new_json(path: Path, value: Any) -> Path:
    """Atomically publish deterministic JSON without replacing existing bytes."""

    destination, parent_fd = _open_output_parent(Path(path), create=True)
    _validate_json_value(value)
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    file_fd: int | None = None
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"refusing to overwrite immutable output: {destination}"
            )
        if not hasattr(os, "O_TMPFILE"):
            raise OSError("immutable publication requires Linux O_TMPFILE support")
        file_fd = os.open(
            ".",
            os.O_WRONLY | os.O_CLOEXEC | os.O_TMPFILE,
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(encoded)
        written = 0
        while written < len(view):
            written += os.write(file_fd, view[written:])
        os.fsync(file_fd)
        os.fchmod(file_fd, 0o444)
        try:
            _link_unnamed_file(file_fd, parent_fd, destination.name)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable output: {destination}"
            ) from exc
        return destination
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _load_json_strict_with_metadata(
    path: Path,
    *,
    max_bytes: int = MAX_SCORECARD_JSON_INPUT_BYTES,
) -> tuple[dict[str, Any], os.stat_result]:
    """Load JSON and return metadata from the descriptor that supplied its bytes."""

    if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_SCORECARD_JSON_INPUT_BYTES:
        raise ValueError("strict JSON byte limit is invalid")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("strict JSON loading requires no-follow file support")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            Path(path),
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{path}: strict JSON input must be a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError(
                f"{path}: strict JSON input exceeds {max_bytes} bytes"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) > max_bytes:
            raise ValueError(
                f"{path}: strict JSON input exceeds {max_bytes} bytes"
            )
        final_metadata = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(encoded) != metadata.st_size or any(
            getattr(metadata, field) != getattr(final_metadata, field)
            for field in stable_fields
        ):
            raise ValueError(f"{path}: strict JSON input changed while being read")
        try:
            source = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: strict JSON input is not UTF-8") from exc
        payload = json.loads(
            source,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except ValueError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load strict JSON from {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    _validate_json_value(payload)
    return payload, final_metadata


def load_json_strict(path: Path) -> dict[str, Any]:
    """Load one bounded regular-file JSON object without following a symlink."""

    payload, _ = _load_json_strict_with_metadata(path)
    return payload


def _record_with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "record_sha256" in result:
        raise ValueError("unfinalized record unexpectedly contains record_sha256")
    result["record_sha256"] = _sha256_json(result)
    return result


def _current_consistency_identities() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Whole-life checkpoint publication is intentionally Linux-only. Keep that
    # dependency behind the artifact-construction boundary so the portable
    # scorecard CLI, help, and deterministic plan remain importable elsewhere.
    return (
        _checkpoint_source_identity(),
        _checkpoint_runtime_identity(),
        _checkpoint_dependency_identity(),
    )


def build_scorecard_artifact(
    plan: ReferenceLifeDevelopmentPlan,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble one immutable aggregate and bind its deterministic summary."""

    ordered_records = sorted(records, key=lambda record: int(record["schedule_index"]))
    summary = summarize_run_records(plan, ordered_records)
    run_order = [
        {
            "schedule_index": spec.schedule_index,
            "environment_kind": spec.environment_kind,
            "arm": spec.arm,
            "seed": spec.seed,
            "lifecycle_id": spec.lifecycle_id,
        }
        for spec in iter_run_specs(plan)
    ]
    source_identity, runtime_identity, dependency_identity = (
        _current_consistency_identities()
    )
    payload: dict[str, Any] = {
        "schema": REFERENCE_LIFE_SCORECARD_ARTIFACT_SCHEMA,
        "schema_version": 1,
        "benchmark": "reference_life_matched_development_scorecard",
        "plan": plan.to_payload(),
        "plan_sha256": plan.plan_sha256,
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "source_identity": source_identity,
        "runtime_identity": runtime_identity,
        "dependency_identity": dependency_identity,
        "identity_scope_note": (
            "consistency binding only; not authenticated execution attestation"
        ),
        "run_order": run_order,
        "runs": [dict(record) for record in ordered_records],
        "summary": summary,
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    return payload


def _require_finite_nonnegative(value: Any, *, path: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{path} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{path} must be a finite nonnegative number")
    return result


def _require_nonnegative_int(value: Any, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a nonnegative integer")
    return value


def _require_finite_number(value: Any, *, path: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _numerically_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _validate_agent_manifest_descriptor(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a complete agent manifest")
    required = {
        "api_version",
        "schema",
        "implementation_id",
        "state_schema",
        "config_sha256",
        "manifest_id",
        "config",
        "observation_spec",
        "action_spec",
        "capabilities",
    }
    if set(value) != required:
        raise ValueError(f"{path} fields do not match the agent-manifest contract")
    if value["api_version"] != REFERENCE_AGENT_API_VERSION:
        raise ValueError(f"{path} API version is unsupported")
    config = value["config"]
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}.config must be an object")
    if canonical_config_sha256(config) != value["config_sha256"]:
        raise ValueError(f"{path} config digest mismatch")
    if not _is_sha256(value["manifest_id"]):
        raise ValueError(f"{path}.manifest_id must be a SHA-256 digest")
    observation_spec = _space_from_descriptor(
        value["observation_spec"], path=f"{path}.observation_spec"
    )
    action_spec = _space_from_descriptor(
        value["action_spec"], path=f"{path}.action_spec"
    )
    capabilities = value["capabilities"]
    if not isinstance(capabilities, Mapping) or set(capabilities) != {
        "dispatch_rebinding"
    }:
        raise ValueError(f"{path}.capabilities fields are invalid")
    reconstructed = AgentManifest.from_config(
        schema=cast(str, value["schema"]),
        implementation_id=cast(str, value["implementation_id"]),
        state_schema=cast(str, value["state_schema"]),
        config=config,
        observation_spec=observation_spec,
        action_spec=action_spec,
        capabilities=AgentCapabilities(
            dispatch_rebinding=capabilities["dispatch_rebinding"]
        ),
    )
    if _agent_manifest_descriptor(reconstructed) != dict(value):
        raise ValueError(f"{path} manifest identity does not recompute")


def _space_from_descriptor(value: Any, *, path: str) -> SpaceSpec:
    if not isinstance(value, Mapping) or set(value) != {
        "kind",
        "shape",
        "dtype",
        "semantic_id",
        "cardinality",
        "low",
        "high",
    }:
        raise ValueError(f"{path} fields do not match a space descriptor")
    shape = value["shape"]
    if not isinstance(shape, list) or any(type(item) is not int for item in shape):
        raise ValueError(f"{path}.shape must be an integer list")
    if value["kind"] == "discrete":
        if value["low"] is not None or value["high"] is not None:
            raise ValueError(f"{path} discrete descriptor cannot carry bounds")
        return SpaceSpec.discrete(
            cardinality=value["cardinality"],
            dtype=value["dtype"],
            semantic_id=value["semantic_id"],
        )
    if value["kind"] == "box":
        low = value["low"]
        high = value["high"]
        if low is not None and not isinstance(low, list):
            raise ValueError(f"{path}.low must be null or a list")
        if high is not None and not isinstance(high, list):
            raise ValueError(f"{path}.high must be null or a list")
        return SpaceSpec.box(
            shape=tuple(shape),
            dtype=value["dtype"],
            low=low,
            high=high,
            semantic_id=value["semantic_id"],
        )
    raise ValueError(f"{path}.kind is unsupported")


def _validate_environment_manifest_descriptor(value: Any, *, path: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "implementation_id",
        "state_schema",
        "config_sha256",
        "manifest_id",
        "config",
        "observation_spec",
        "action_spec",
        "max_executions",
    }:
        raise ValueError(f"{path} fields do not match an environment manifest")
    config = value["config"]
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}.config must be an object")
    reconstructed = ReferenceEnvironmentManifest.from_config(
        implementation_id=cast(str, value["implementation_id"]),
        state_schema=cast(str, value["state_schema"]),
        config=config,
        observation_spec=_space_from_descriptor(
            value["observation_spec"], path=f"{path}.observation_spec"
        ),
        action_spec=_space_from_descriptor(
            value["action_spec"], path=f"{path}.action_spec"
        ),
        max_executions=cast(int, value["max_executions"]),
    )
    if reconstructed.descriptor() != dict(value):
        raise ValueError(f"{path} manifest identity does not recompute")


def _validate_resolved_components(
    resolved: Mapping[str, Any],
    *,
    plan: ReferenceLifeDevelopmentPlan,
    spec: ScorecardRunSpec,
    path: str,
) -> None:
    agent_manifest = resolved["agent_manifest"]
    _validate_agent_manifest_descriptor(
        agent_manifest, path=f"{path}.agent_manifest"
    )
    environment_manifest = resolved["environment_manifest"]
    _validate_environment_manifest_descriptor(
        environment_manifest, path=f"{path}.environment_manifest"
    )
    assert isinstance(agent_manifest, Mapping)
    assert isinstance(environment_manifest, Mapping)
    environment_config = environment_manifest["config"]
    assert isinstance(environment_config, Mapping)
    if environment_config.get("environment_kind") != spec.environment_kind:
        raise ValueError(f"{path} environment kind differs from the schedule")
    if agent_manifest["observation_spec"] != environment_manifest["observation_spec"]:
        raise ValueError(f"{path} agent/environment observation specs differ")
    if agent_manifest["action_spec"] != environment_manifest["action_spec"]:
        raise ValueError(f"{path} agent/environment action specs differ")

    life_config = resolved["life_config"]
    if not isinstance(life_config, Mapping):
        raise ValueError(f"{path} lacks a complete life config")
    if _sha256_json(life_config) != resolved["life_config_sha256"]:
        raise ValueError(f"{path} life config digest mismatch")
    protocol = plan.protocol(spec.environment_kind)
    expected_scalars = {
        "lifecycle_id": spec.lifecycle_id,
        "seed": spec.seed,
        "max_accepted_events": protocol["horizon"],
        "rng_schedule": REFERENCE_LIFE_RNG_SCHEDULE,
    }
    for field, expected in expected_scalars.items():
        if life_config.get(field) != expected:
            raise ValueError(f"{path}.life_config.{field} differs from the schedule")
    life_agent = life_config.get("agent")
    life_environment = life_config.get("environment")
    if not isinstance(life_agent, Mapping) or not isinstance(life_environment, Mapping):
        raise ValueError(f"{path}.life_config lacks complete component bindings")
    if life_agent.get("manifest_id") != agent_manifest["manifest_id"]:
        raise ValueError(f"{path}.life_config binds another agent manifest")
    if life_environment.get("manifest_id") != environment_manifest["manifest_id"]:
        raise ValueError(f"{path}.life_config binds another environment manifest")

    canonical_runner = build_scorecard_runner(plan, spec)
    canonical_resolved = _resolved_components(plan, spec, canonical_runner)
    if not _json_exact_equal(dict(resolved), canonical_resolved):
        raise ValueError(
            f"{path} does not match the canonical resolved components for the "
            "scheduled arm and exact environment definition"
        )


def _validate_resource_payload(resource: Any, *, arm: str, path: str) -> None:
    if not isinstance(resource, Mapping):
        raise ValueError(f"{path} must be an object")
    expected_method = (
        "prototype_control_parameter_tree"
        if arm in ("prototype", "prototype_frozen")
        else (
            "known_sarsa_parameter_fields"
            if arm in ("differential_sarsa", "sarsa")
            else "floating_jax_pytree_leaves_upper_bound"
        )
    )
    required = {
        "persistent_jax_array_leaves",
        "persistent_jax_array_scalar_count",
        "persistent_jax_array_bytes",
        "persistent_static_numeric_bytes",
        "persistent_numeric_bytes_total",
        "persistent_state_measurement",
        "trainable_scalar_count_estimate",
        "trainable_scalar_count_method",
        "hardware_peak_memory_measured",
    }
    if expected_method != "floating_jax_pytree_leaves_upper_bound":
        required.add("trainable_parameter_tree_scalar_count")
    if set(resource) != required:
        raise ValueError(f"{path} fields do not match the resource contract")
    if resource["persistent_state_measurement"] != (
        "numeric_jax_and_numpy_pytree_leaves"
    ):
        raise ValueError(f"{path}.persistent_state_measurement is unsupported")
    if resource["trainable_scalar_count_method"] != expected_method:
        raise ValueError(f"{path}.trainable_scalar_count_method differs from the arm")
    if resource["hardware_peak_memory_measured"] is not False:
        raise ValueError(f"{path} must not claim a hardware peak measurement")
    leaves = _require_nonnegative_int(
        resource["persistent_jax_array_leaves"],
        path=f"{path}.persistent_jax_array_leaves",
    )
    elements = _require_nonnegative_int(
        resource["persistent_jax_array_scalar_count"],
        path=f"{path}.persistent_jax_array_scalar_count",
    )
    byte_count = _require_nonnegative_int(
        resource["persistent_jax_array_bytes"],
        path=f"{path}.persistent_jax_array_bytes",
    )
    static_bytes = _require_nonnegative_int(
        resource["persistent_static_numeric_bytes"],
        path=f"{path}.persistent_static_numeric_bytes",
    )
    total_bytes = _require_nonnegative_int(
        resource["persistent_numeric_bytes_total"],
        path=f"{path}.persistent_numeric_bytes_total",
    )
    if total_bytes != byte_count + static_bytes:
        raise ValueError(f"{path} persistent numeric byte total is inconsistent")
    trainable = _require_nonnegative_int(
        resource["trainable_scalar_count_estimate"],
        path=f"{path}.trainable_scalar_count_estimate",
    )
    if leaves == 0 and (elements != 0 or byte_count != 0):
        raise ValueError(f"{path} reports numeric storage without an array leaf")
    if (elements == 0) != (byte_count == 0):
        raise ValueError(f"{path} scalar and byte counts disagree about empty storage")
    if elements > 0 and byte_count < elements:
        raise ValueError(f"{path} byte count is smaller than its numeric scalar count")
    if trainable > elements:
        raise ValueError(f"{path} trainable estimate exceeds persistent numeric state")
    if expected_method != "floating_jax_pytree_leaves_upper_bound":
        parameter_elements = _require_nonnegative_int(
            resource["trainable_parameter_tree_scalar_count"],
            path=f"{path}.trainable_parameter_tree_scalar_count",
        )
        if parameter_elements > elements or trainable > parameter_elements:
            raise ValueError(f"{path} parameter-tree counts exceed persistent state")


def _reward_lattice_matches(
    total: float,
    *,
    event_count: int,
    reward_values: Sequence[float],
) -> bool:
    """Return whether a total is reachable from the selected discrete rewards."""

    values = sorted({float(np.float32(value)) for value in reward_values})
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        return False
    positive = [value for value in values if value > 0.0]
    tolerance = float(
        max(1e-12, event_count * float(np.spacing(max(1.0, abs(total)))) * 4.0)
    )
    if not positive:
        return abs(total) <= tolerance
    if len(positive) == 1:
        count = round(total / positive[0])
        return 0 <= count <= event_count and math.isclose(
            total,
            count * positive[0],
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    if len(positive) == 2:
        low, high = positive
        maximum_high = min(event_count, max(0, int(math.floor(total / high)) + 1))
        for high_count in range(maximum_high + 1):
            residual = total - high_count * high
            low_count = round(residual / low)
            if (
                0 <= low_count <= event_count - high_count
                and math.isclose(
                    total,
                    high_count * high + low_count * low,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
            ):
                return True
    return False


def _validate_metric_window(
    window: Any,
    *,
    event_count: int,
    oracle_reward: float,
    minimum_reward: float,
    maximum_reward: float,
    reward_values: Sequence[float],
    path: str,
) -> float:
    required = {"event_count", "reward_sum", "mean_reward", "mean_oracle_regret"}
    if not isinstance(window, Mapping) or set(window) != required:
        raise ValueError(f"{path} fields do not match the metric-window contract")
    if window["event_count"] != event_count or type(window["event_count"]) is not int:
        raise ValueError(f"{path}.event_count differs from the fixed window")
    reward_sum = _require_finite_number(window["reward_sum"], path=f"{path}.reward_sum")
    mean_reward = _require_finite_number(
        window["mean_reward"], path=f"{path}.mean_reward"
    )
    mean_regret = _require_finite_number(
        window["mean_oracle_regret"], path=f"{path}.mean_oracle_regret"
    )
    if not _numerically_equal(mean_reward, reward_sum / event_count):
        raise ValueError(f"{path}.mean_reward is inconsistent")
    if not _numerically_equal(mean_regret, oracle_reward - mean_reward):
        raise ValueError(f"{path}.mean_oracle_regret is inconsistent")
    tolerance = 1e-9 * max(1.0, abs(reward_sum))
    if not (
        minimum_reward * event_count - tolerance
        <= reward_sum
        <= maximum_reward * event_count + tolerance
    ):
        raise ValueError(f"{path}.reward_sum is outside the exact environment range")
    if not _reward_lattice_matches(
        reward_sum,
        event_count=event_count,
        reward_values=reward_values,
    ):
        raise ValueError(f"{path}.reward_sum is outside the exact reward lattice")
    return reward_sum


def _validate_completed_outcome(
    outcome: Any,
    *,
    environment: str,
    arm: str,
    protocol: Mapping[str, Any],
    expected_initial_resource: Mapping[str, Any],
    path: str,
) -> None:
    if not isinstance(outcome, Mapping):
        raise ValueError(f"{path} must be an object")
    required = {
        "summary_mode",
        "configured_horizon",
        "accepted_events",
        "reward_sum",
        "mean_reward",
        "oracle_reward_sum",
        "regret_sum",
        "parameter_change_events",
        "phase_event_counts",
        "phase_reward_sums",
        "windows",
        "high_end_visit_count",
        "high_end_visit_rate",
        "environment_rng_cursor",
        "transcript_sha256",
        "resources",
        "parameter_change_check",
    }
    if set(outcome) != required:
        raise ValueError(f"{path} fields do not match the completed-outcome contract")
    horizon = protocol["horizon"]
    if (
        type(outcome["configured_horizon"]) is not int
        or type(outcome["accepted_events"]) is not int
        or outcome["configured_horizon"] != horizon
        or outcome["accepted_events"] != horizon
    ):
        raise ValueError(f"{path} does not reach the fixed horizon")
    if (
        type(outcome["environment_rng_cursor"]) is not int
        or outcome["environment_rng_cursor"] != horizon
    ):
        raise ValueError(f"{path} environment RNG cursor is not matched to the horizon")
    if outcome["summary_mode"] != "streaming_o1_no_retained_events":
        raise ValueError(f"{path} did not use the streaming summary")
    reward_sum = _require_finite_number(
        outcome["reward_sum"], path=f"{path}.reward_sum"
    )
    oracle_reward_sum = _require_finite_number(
        outcome["oracle_reward_sum"], path=f"{path}.oracle_reward_sum"
    )
    mean_reward = _require_finite_number(
        outcome["mean_reward"], path=f"{path}.mean_reward"
    )
    regret_sum = _require_finite_number(
        outcome["regret_sum"], path=f"{path}.regret_sum"
    )
    if not _numerically_equal(mean_reward, reward_sum / horizon):
        raise ValueError(f"{path}.mean_reward is inconsistent")
    if not _numerically_equal(regret_sum, oracle_reward_sum - reward_sum):
        raise ValueError(f"{path}.regret_sum is inconsistent")
    changes = outcome["parameter_change_events"]
    if type(changes) is not int or not 0 <= changes <= horizon:
        raise ValueError(f"{path}.parameter_change_events is invalid")
    if outcome["parameter_change_check"] != parameter_change_check(arm, changes):
        raise ValueError(f"{path} parameter-change check is inconsistent")
    if not _is_sha256(outcome["transcript_sha256"]):
        raise ValueError(f"{path}.transcript_sha256 must be a digest")
    resources = outcome["resources"]
    if not isinstance(resources, Mapping) or set(resources) != {"initial", "final"}:
        raise ValueError(f"{path}.resources must bind initial and final state")
    for label in ("initial", "final"):
        _validate_resource_payload(
            resources[label], arm=arm, path=f"{path}.resources.{label}"
        )
    if not _json_exact_equal(dict(resources["initial"]), dict(expected_initial_resource)):
        raise ValueError(f"{path}.resources.initial differs from the canonical agent state")
    if not _json_exact_equal(dict(resources["final"]), dict(resources["initial"])):
        raise ValueError(f"{path}.resources reports persistent-state growth or shrinkage")

    windows = outcome["windows"]
    if not isinstance(windows, Mapping):
        raise ValueError(f"{path}.windows must be an object")
    phase_counts = outcome["phase_event_counts"]
    phase_rewards = outcome["phase_reward_sums"]
    if (
        not isinstance(phase_counts, list)
        or len(phase_counts) != 2
        or any(type(value) is not int or value < 0 for value in phase_counts)
        or sum(phase_counts) != horizon
    ):
        raise ValueError(f"{path}.phase_event_counts is inconsistent")
    if not isinstance(phase_rewards, list) or len(phase_rewards) != 2:
        raise ValueError(f"{path}.phase_reward_sums is inconsistent")
    phase_reward_values = [
        _require_finite_number(value, path=f"{path}.phase_reward_sums[{index}]")
        for index, value in enumerate(phase_rewards)
    ]
    if not math.isclose(
        math.fsum(phase_reward_values),
        reward_sum,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path}.phase_reward_sums is inconsistent")

    window_reward_sums: list[float] = []
    if environment == "switching_two_state":
        if set(windows) != {"initial_a", "first_b", "return_a"}:
            raise ValueError(f"{path} lacks the fixed A/B/A windows")
        phase_length = protocol["phase_length"]
        expected_phase_counts = [
            sum(
                min(phase_length, horizon - start)
                for start in range(0, horizon, phase_length)
                if (start // phase_length) % 2 == phase
            )
            for phase in (0, 1)
        ]
        if phase_counts != expected_phase_counts:
            raise ValueError(f"{path}.phase_event_counts violates the switching schedule")
        switching_config = SwitchingTwoStateConfig(  # type: ignore[call-arg]
            phase_length=phase_length,
            payoffs_a=tuple(tuple(row) for row in protocol["payoffs_a"]),
            payoffs_b=tuple(tuple(row) for row in protocol["payoffs_b"]),
        )
        switching = SwitchingTwoStateMDP(switching_config)
        oracle_rewards = [switching.optimal_average_reward(phase) for phase in (0, 1)]
        phase_payoffs = (protocol["payoffs_a"], protocol["payoffs_b"])
        for phase in (0, 1):
            flattened = [float(value) for row in phase_payoffs[phase] for value in row]
            tolerance = 1e-9 * max(1.0, abs(phase_reward_values[phase]))
            if not (
                min(flattened) * phase_counts[phase] - tolerance
                <= phase_reward_values[phase]
                <= max(flattened) * phase_counts[phase] + tolerance
            ):
                raise ValueError(
                    f"{path}.phase_reward_sums[{phase}] is outside the payoff range"
                )
            if not _reward_lattice_matches(
                phase_reward_values[phase],
                event_count=phase_counts[phase],
                reward_values=flattened,
            ):
                raise ValueError(
                    f"{path}.phase_reward_sums[{phase}] is outside the reward lattice"
                )
        for name, phase in (("initial_a", 0), ("first_b", 1), ("return_a", 0)):
            flattened = [float(value) for row in phase_payoffs[phase] for value in row]
            window_reward_sums.append(
                _validate_metric_window(
                    windows[name],
                    event_count=protocol["post_switch_window"],
                    oracle_reward=oracle_rewards[phase],
                    minimum_reward=min(flattened),
                    maximum_reward=max(flattened),
                    reward_values=flattened,
                    path=f"{path}.windows.{name}",
                )
            )
        if (
            window_reward_sums[0] + window_reward_sums[2]
            > phase_reward_values[0] + 1e-9
            or window_reward_sums[1] > phase_reward_values[1] + 1e-9
        ):
            raise ValueError(f"{path}.windows exceed their switching-phase rewards")
        if outcome["high_end_visit_count"] is not None or outcome[
            "high_end_visit_rate"
        ] is not None:
            raise ValueError(f"{path} switching run must not claim RiverSwim visits")
    elif environment == "riverswim":
        if set(windows) != {"early", "late"}:
            raise ValueError(f"{path} lacks the fixed RiverSwim windows")
        if phase_counts != [horizon, 0]:
            raise ValueError(f"{path}.phase_event_counts violates stationary semantics")
        river_config = RiverSwimConfig(  # type: ignore[call-arg]
            n_states=protocol["n_states"],
            p_right_up=protocol["p_right_up"],
            p_right_down=protocol["p_right_down"],
            reward_left=protocol["reward_left"],
            reward_right=protocol["reward_right"],
            initial_state=protocol["initial_state"],
        )
        river = RiverSwimMDP(river_config)
        oracle_rewards = [river.optimal_average_reward(), 0.0]
        possible_rewards = (0.0, protocol["reward_left"], protocol["reward_right"])
        minimum_reward = min(possible_rewards)
        maximum_reward = max(possible_rewards)
        tolerance = 1e-9 * max(1.0, abs(phase_reward_values[0]))
        if not (
            minimum_reward * horizon - tolerance
            <= phase_reward_values[0]
            <= maximum_reward * horizon + tolerance
        ) or phase_reward_values[1] != 0.0:
            raise ValueError(f"{path}.phase_reward_sums violates stationary rewards")
        if not _reward_lattice_matches(
            phase_reward_values[0],
            event_count=horizon,
            reward_values=possible_rewards,
        ):
            raise ValueError(f"{path}.phase_reward_sums violates the reward lattice")
        for name, expected_count in (
            ("early", protocol["early_window"]),
            ("late", protocol["late_window"]),
        ):
            window_reward_sums.append(
                _validate_metric_window(
                    windows[name],
                    event_count=expected_count,
                    oracle_reward=oracle_rewards[0],
                    minimum_reward=minimum_reward,
                    maximum_reward=maximum_reward,
                    reward_values=possible_rewards,
                    path=f"{path}.windows.{name}",
                )
            )
        if math.fsum(window_reward_sums) > phase_reward_values[0] + 1e-9:
            raise ValueError(f"{path}.windows exceed the stationary reward sum")
        visits = outcome["high_end_visit_count"]
        if type(visits) is not int or not 0 <= visits <= horizon:
            raise ValueError(f"{path}.high_end_visit_count is invalid")
        rate = _require_finite_number(
            outcome["high_end_visit_rate"], path=f"{path}.high_end_visit_rate"
        )
        if not _numerically_equal(rate, visits / horizon):
            raise ValueError(f"{path}.high_end_visit_rate is inconsistent")
    else:
        raise ValueError(f"{path} has an unsupported environment")

    expected_oracle_sum = math.fsum(
        count * oracle for count, oracle in zip(phase_counts, oracle_rewards, strict=True)
    )
    if not _numerically_equal(oracle_reward_sum, expected_oracle_sum):
        raise ValueError(f"{path}.oracle_reward_sum differs from the exact environment")
    all_reward_values = (
        [
            float(value)
            for matrix in (protocol["payoffs_a"], protocol["payoffs_b"])
            for row in matrix
            for value in row
        ]
        if environment == "switching_two_state"
        else [0.0, protocol["reward_left"], protocol["reward_right"]]
    )
    if not _reward_lattice_matches(
        reward_sum,
        event_count=horizon,
        reward_values=all_reward_values,
    ):
        raise ValueError(f"{path}.reward_sum is outside the exact reward lattice")


def _partial_window_count(start: int, stop: int, accepted: int) -> int:
    return max(0, min(stop, accepted) - start)


def _validate_partial_outcome(
    partial: Any,
    *,
    accepted: int,
    environment: str,
    protocol: Mapping[str, Any],
    path: str,
) -> None:
    required = {
        "summary_mode",
        "configured_horizon",
        "accepted_events",
        "reward_sum",
        "mean_reward",
        "oracle_reward_sum",
        "regret_sum",
        "parameter_change_events",
        "phase_event_counts",
        "phase_reward_sums",
        "windows",
        "high_end_visit_count",
        "high_end_visit_rate",
    }
    if not isinstance(partial, Mapping) or set(partial) != required:
        raise ValueError(f"{path} fields do not match the partial-outcome contract")
    horizon = protocol["horizon"]
    if (
        partial["summary_mode"] != "streaming_o1_no_retained_events"
        or type(partial["configured_horizon"]) is not int
        or partial["configured_horizon"] != horizon
        or type(partial["accepted_events"]) is not int
        or partial["accepted_events"] != accepted
    ):
        raise ValueError(f"{path} counters do not match the failed shard")
    reward_sum = _require_finite_number(partial["reward_sum"], path=f"{path}.reward_sum")
    oracle_sum = _require_finite_number(
        partial["oracle_reward_sum"], path=f"{path}.oracle_reward_sum"
    )
    regret_sum = _require_finite_number(
        partial["regret_sum"], path=f"{path}.regret_sum"
    )
    mean = partial["mean_reward"]
    if accepted == 0:
        if mean is not None or any(value != 0.0 for value in (reward_sum, oracle_sum, regret_sum)):
            raise ValueError(f"{path} zero-event arithmetic is inconsistent")
    else:
        mean_value = _require_finite_number(mean, path=f"{path}.mean_reward")
        if not _numerically_equal(mean_value, reward_sum / accepted):
            raise ValueError(f"{path}.mean_reward is inconsistent")
    if not _numerically_equal(regret_sum, oracle_sum - reward_sum):
        raise ValueError(f"{path}.regret_sum is inconsistent")
    changes = partial["parameter_change_events"]
    if type(changes) is not int or not 0 <= changes <= accepted:
        raise ValueError(f"{path}.parameter_change_events is invalid")

    phase_counts = partial["phase_event_counts"]
    phase_rewards = partial["phase_reward_sums"]
    if (
        not isinstance(phase_counts, list)
        or len(phase_counts) != 2
        or any(type(value) is not int or value < 0 for value in phase_counts)
        or sum(phase_counts) != accepted
        or not isinstance(phase_rewards, list)
        or len(phase_rewards) != 2
    ):
        raise ValueError(f"{path} phase summaries are malformed")
    phase_reward_values = [
        _require_finite_number(value, path=f"{path}.phase_reward_sums[{index}]")
        for index, value in enumerate(phase_rewards)
    ]
    if not _numerically_equal(math.fsum(phase_reward_values), reward_sum):
        raise ValueError(f"{path}.phase_reward_sums do not sum to reward_sum")
    windows = partial["windows"]
    if not isinstance(windows, Mapping):
        raise ValueError(f"{path}.windows must be an object")

    if environment == "switching_two_state":
        phase_length = protocol["phase_length"]
        cycles, remainder = divmod(accepted, 2 * phase_length)
        expected_counts = [
            cycles * phase_length + min(remainder, phase_length),
            cycles * phase_length + max(0, remainder - phase_length),
        ]
        if phase_counts != expected_counts:
            raise ValueError(f"{path}.phase_event_counts violates the switching schedule")
        kernel = SwitchingTwoStateMDP(
            SwitchingTwoStateConfig(  # type: ignore[call-arg]
                phase_length=phase_length,
                payoffs_a=tuple(tuple(row) for row in protocol["payoffs_a"]),
                payoffs_b=tuple(tuple(row) for row in protocol["payoffs_b"]),
            )
        )
        payoffs = (protocol["payoffs_a"], protocol["payoffs_b"])
        oracle_rewards = [kernel.optimal_average_reward(phase) for phase in (0, 1)]
        for phase in (0, 1):
            rewards = [float(value) for row in payoffs[phase] for value in row]
            if not _reward_lattice_matches(
                phase_reward_values[phase],
                event_count=phase_counts[phase],
                reward_values=rewards,
            ):
                raise ValueError(f"{path}.phase_reward_sums[{phase}] violates rewards")
        if set(windows) != {"initial_a", "first_b", "return_a"}:
            raise ValueError(f"{path}.windows does not match the A/B/A contract")
        window = protocol["post_switch_window"]
        for name, start, phase in (
            ("initial_a", 0, 0),
            ("first_b", phase_length, 1),
            ("return_a", 2 * phase_length, 0),
        ):
            expected_count = _partial_window_count(start, start + window, accepted)
            value = windows[name]
            if expected_count == 0:
                if value is not None:
                    raise ValueError(f"{path}.windows.{name} must be empty")
            else:
                rewards = [float(item) for row in payoffs[phase] for item in row]
                _validate_metric_window(
                    value,
                    event_count=expected_count,
                    oracle_reward=oracle_rewards[phase],
                    minimum_reward=min(rewards),
                    maximum_reward=max(rewards),
                    reward_values=rewards,
                    path=f"{path}.windows.{name}",
                )
        if partial["high_end_visit_count"] is not None or partial[
            "high_end_visit_rate"
        ] is not None:
            raise ValueError(f"{path} switching partial cannot claim RiverSwim visits")
    elif environment == "riverswim":
        if phase_counts != [accepted, 0] or phase_reward_values[1] != 0.0:
            raise ValueError(f"{path} violates stationary RiverSwim semantics")
        river = RiverSwimMDP(
            RiverSwimConfig(  # type: ignore[call-arg]
                n_states=protocol["n_states"],
                p_right_up=protocol["p_right_up"],
                p_right_down=protocol["p_right_down"],
                reward_left=protocol["reward_left"],
                reward_right=protocol["reward_right"],
                initial_state=protocol["initial_state"],
            )
        )
        oracle_rewards = [river.optimal_average_reward(), 0.0]
        river_rewards = (0.0, protocol["reward_left"], protocol["reward_right"])
        if not _reward_lattice_matches(
            phase_reward_values[0], event_count=accepted, reward_values=river_rewards
        ):
            raise ValueError(f"{path}.phase_reward_sums violates RiverSwim rewards")
        if set(windows) != {"early", "late"}:
            raise ValueError(f"{path}.windows does not match the RiverSwim contract")
        for name, start, stop in (
            ("early", 0, protocol["early_window"]),
            ("late", horizon - protocol["late_window"], horizon),
        ):
            expected_count = _partial_window_count(start, stop, accepted)
            value = windows[name]
            if expected_count == 0:
                if value is not None:
                    raise ValueError(f"{path}.windows.{name} must be empty")
            else:
                _validate_metric_window(
                    value,
                    event_count=expected_count,
                    oracle_reward=oracle_rewards[0],
                    minimum_reward=min(river_rewards),
                    maximum_reward=max(river_rewards),
                    reward_values=river_rewards,
                    path=f"{path}.windows.{name}",
                )
        visits = partial["high_end_visit_count"]
        if type(visits) is not int or not 0 <= visits <= accepted:
            raise ValueError(f"{path}.high_end_visit_count is invalid")
        rate = partial["high_end_visit_rate"]
        if accepted == 0:
            if rate is not None:
                raise ValueError(f"{path}.high_end_visit_rate must be empty")
        else:
            rate_value = _require_finite_number(rate, path=f"{path}.high_end_visit_rate")
            if not _numerically_equal(rate_value, visits / accepted):
                raise ValueError(f"{path}.high_end_visit_rate is inconsistent")
    else:
        raise ValueError(f"{path} has an unsupported environment")
    expected_oracle = math.fsum(
        count * oracle
        for count, oracle in zip(phase_counts, oracle_rewards, strict=True)
    )
    if not _numerically_equal(oracle_sum, expected_oracle):
        raise ValueError(f"{path}.oracle_reward_sum is inconsistent")


def _validate_run_record(
    plan: ReferenceLifeDevelopmentPlan,
    record: Any,
    expected_spec: ScorecardRunSpec,
    *,
    path: str,
    consistency_identities: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{path} must be an object")
    required = {
        "schema",
        "plan_sha256",
        "schedule_index",
        "environment_kind",
        "arm",
        "seed",
        "lifecycle_id",
        "evidence_policy",
        "source_identity",
        "runtime_identity",
        "dependency_identity",
        "status",
        "failure",
        "resolved",
        "outcome",
        "partial_outcome",
        "telemetry",
        "record_sha256",
    }
    if set(record) != required:
        raise ValueError(f"{path} fields do not match the run-record contract")
    expected_identity = (
        expected_spec.schedule_index,
        expected_spec.environment_kind,
        expected_spec.arm,
        expected_spec.seed,
        expected_spec.lifecycle_id,
    )
    observed_identity = (
        record["schedule_index"],
        record["environment_kind"],
        record["arm"],
        record["seed"],
        record["lifecycle_id"],
    )
    if (
        type(record["schedule_index"]) is not int
        or type(record["seed"]) is not int
        or type(record["environment_kind"]) is not str
        or type(record["arm"]) is not str
        or type(record["lifecycle_id"]) is not str
        or observed_identity != expected_identity
    ):
        raise ValueError(f"{path} is outside the canonical cyclic schedule")
    if record["schema"] != REFERENCE_LIFE_SCORECARD_RUN_SCHEMA:
        raise ValueError(f"{path} has an unsupported schema")
    if record["plan_sha256"] != plan.plan_sha256:
        raise ValueError(f"{path} belongs to another plan")
    if not _json_exact_equal(record["evidence_policy"], NONPROMOTING_POLICY):
        raise ValueError(f"{path} must remain permanently nonpromoting")
    source_identity, runtime_identity, dependency_identity = consistency_identities
    if not _json_exact_equal(record["source_identity"], source_identity):
        raise ValueError(f"{path} source identity differs from the current source tree")
    if not _json_exact_equal(record["runtime_identity"], runtime_identity):
        raise ValueError(f"{path} runtime identity differs from the current runtime")
    if not _json_exact_equal(record["dependency_identity"], dependency_identity):
        raise ValueError(f"{path} dependency identity differs from current dependencies")
    if record["record_sha256"] != _digest_excluding(record, "record_sha256"):
        raise ValueError(f"{path} content digest mismatch")

    telemetry = record["telemetry"]
    telemetry_fields = {
        "policy",
        "setup_seconds",
        "cold_step_seconds",
        "warmed_step_seconds_total",
        "warmed_step_count",
        "warmed_step_seconds_mean",
        "total_seconds",
    }
    if not isinstance(telemetry, Mapping) or set(telemetry) != telemetry_fields:
        raise ValueError(f"{path}.telemetry fields do not match the telemetry contract")
    if not _json_exact_equal(telemetry["policy"], TELEMETRY_POLICY):
        raise ValueError(f"{path}.telemetry is not clearly telemetry-only")
    optional_durations: dict[str, float | None] = {}
    for field in ("setup_seconds", "cold_step_seconds"):
        value = telemetry[field]
        optional_durations[field] = (
            None
            if value is None
            else _require_finite_nonnegative(
                value, path=f"{path}.telemetry.{field}"
            )
        )
    warmed_total = _require_finite_nonnegative(
        telemetry["warmed_step_seconds_total"],
        path=f"{path}.telemetry.warmed_step_seconds_total",
    )
    total_seconds = _require_finite_nonnegative(
        telemetry["total_seconds"], path=f"{path}.telemetry.total_seconds"
    )
    warmed_count = _require_nonnegative_int(
        telemetry["warmed_step_count"], path=f"{path}.telemetry.warmed_step_count"
    )
    warmed_mean_value = telemetry["warmed_step_seconds_mean"]
    if warmed_count == 0:
        if warmed_total != 0.0 or warmed_mean_value is not None:
            raise ValueError(f"{path}.telemetry warmed timing is inconsistent")
    else:
        warmed_mean = _require_finite_nonnegative(
            warmed_mean_value, path=f"{path}.telemetry.warmed_step_seconds_mean"
        )
        if not _numerically_equal(warmed_mean, warmed_total / warmed_count):
            raise ValueError(f"{path}.telemetry warmed timing is inconsistent")
    timed_components = warmed_total + math.fsum(
        value for value in optional_durations.values() if value is not None
    )
    if total_seconds + 1e-12 < timed_components:
        raise ValueError(f"{path}.telemetry.total_seconds is smaller than timed work")

    resolved = record["resolved"]
    if not isinstance(resolved, Mapping):
        raise ValueError(f"{path}.resolved must be an object")
    expected_resolved_fields = {
        "arm_definition",
        "agent_manifest",
        "environment_manifest",
        "life_config",
        "life_config_sha256",
    }
    if set(resolved) != expected_resolved_fields:
        raise ValueError(f"{path}.resolved fields are incomplete")
    if not _json_exact_equal(
        resolved["arm_definition"], plan.arm_definition(expected_spec.arm)
    ):
        raise ValueError(f"{path} arm definition differs from the plan")

    status = record["status"]
    if status == "completed":
        if record["failure"] is not None or record["partial_outcome"] is not None:
            raise ValueError(f"{path} completed record carries failure data")
        horizon = plan.protocol(expected_spec.environment_kind)["horizon"]
        if (
            optional_durations["setup_seconds"] is None
            or optional_durations["cold_step_seconds"] is None
            or warmed_count != horizon - 1
        ):
            raise ValueError(f"{path}.telemetry does not cover every completed step")
        _validate_resolved_components(
            resolved,
            plan=plan,
            spec=expected_spec,
            path=f"{path}.resolved",
        )
        _validate_completed_outcome(
            record["outcome"],
            environment=expected_spec.environment_kind,
            arm=expected_spec.arm,
            protocol=plan.protocol(expected_spec.environment_kind),
            expected_initial_resource=_canonical_initial_resource_payload(
                expected_spec.environment_kind,
                expected_spec.arm,
            ),
            path=f"{path}.outcome",
        )
    elif status == "failed":
        if record["outcome"] is not None:
            raise ValueError(f"{path} failed record cannot carry a completed outcome")
        failure = record["failure"]
        if not isinstance(failure, Mapping) or set(failure) != {
            "stage",
            "type",
            "message",
            "accepted_events",
        }:
            raise ValueError(f"{path} failure record is incomplete")
        if failure["stage"] not in {"build", "init", "step"}:
            raise ValueError(f"{path} failure stage is unsupported")
        if type(failure["type"]) is not str or not failure["type"]:
            raise ValueError(f"{path} failure type must be nonempty")
        if type(failure["message"]) is not str or not failure["message"]:
            raise ValueError(f"{path} failure message must be nonempty")
        if type(failure["accepted_events"]) is not int or failure["accepted_events"] < 0:
            raise ValueError(f"{path} failure accepted-event count is invalid")
        horizon = plan.protocol(expected_spec.environment_kind)["horizon"]
        if failure["accepted_events"] > horizon:
            raise ValueError(f"{path} failure accepted-event count exceeds the horizon")
        partial = record["partial_outcome"]
        _validate_partial_outcome(
            partial,
            accepted=failure["accepted_events"],
            environment=expected_spec.environment_kind,
            protocol=plan.protocol(expected_spec.environment_kind),
            path=f"{path}.partial_outcome",
        )
        if failure["stage"] in {"build", "init"} and failure["accepted_events"] != 0:
            raise ValueError(f"{path} pre-step failure cannot claim accepted events")
        if failure["stage"] == "build":
            if any(
                resolved[field] is not None
                for field in (
                    "agent_manifest",
                    "environment_manifest",
                    "life_config",
                    "life_config_sha256",
                )
            ):
                raise ValueError(f"{path} build failure carries partial live bindings")
        else:
            _validate_resolved_components(
                resolved,
                plan=plan,
                spec=expected_spec,
                path=f"{path}.resolved",
            )
    else:
        raise ValueError(f"{path} status must be completed or failed")


def validate_scorecard_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate and recompute one scorecard aggregate."""

    required = {
        "schema",
        "schema_version",
        "benchmark",
        "plan",
        "plan_sha256",
        "evidence_policy",
        "source_identity",
        "runtime_identity",
        "dependency_identity",
        "identity_scope_note",
        "run_order",
        "runs",
        "summary",
        "artifact_sha256",
    }
    if set(payload) != required:
        raise ValueError("artifact fields do not match the scorecard schema")
    if (
        payload["schema"] != REFERENCE_LIFE_SCORECARD_ARTIFACT_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["benchmark"] != "reference_life_matched_development_scorecard"
    ):
        raise ValueError("artifact schema identity is unsupported")
    if not _json_exact_equal(payload["evidence_policy"], NONPROMOTING_POLICY):
        raise ValueError("artifact must remain permanently nonpromoting")
    consistency_identities = _current_consistency_identities()
    source_identity, runtime_identity, dependency_identity = consistency_identities
    if not _json_exact_equal(payload["source_identity"], source_identity):
        raise ValueError("artifact source identity differs from the current source tree")
    if not _json_exact_equal(payload["runtime_identity"], runtime_identity):
        raise ValueError("artifact runtime identity differs from the current runtime")
    if not _json_exact_equal(payload["dependency_identity"], dependency_identity):
        raise ValueError("artifact dependency identity differs from current dependencies")
    if payload["identity_scope_note"] != (
        "consistency binding only; not authenticated execution attestation"
    ):
        raise ValueError("artifact identity scope note is misleading")
    plan_value = payload["plan"]
    if not isinstance(plan_value, Mapping):
        raise ValueError("artifact plan must be an object")
    plan = ReferenceLifeDevelopmentPlan.from_payload(plan_value)
    if payload["plan_sha256"] != plan.plan_sha256:
        raise ValueError("artifact plan digest mismatch")
    specs = iter_run_specs(plan)
    expected_order = [
        {
            "schedule_index": spec.schedule_index,
            "environment_kind": spec.environment_kind,
            "arm": spec.arm,
            "seed": spec.seed,
            "lifecycle_id": spec.lifecycle_id,
        }
        for spec in specs
    ]
    if not _json_exact_equal(payload["run_order"], expected_order):
        raise ValueError("artifact run order is not the fixed cyclic schedule")
    runs = payload["runs"]
    if not isinstance(runs, list) or len(runs) != len(specs):
        raise ValueError("artifact must retain exactly one record per scheduled run")
    for index, (record, spec) in enumerate(zip(runs, specs, strict=True)):
        _validate_run_record(
            plan,
            record,
            spec,
            path=f"$.runs[{index}]",
            consistency_identities=consistency_identities,
        )
    recomputed_summary = _summarize_validated_run_records(
        plan,
        cast(list[Mapping[str, Any]], runs),
    )
    if not _json_exact_equal(payload["summary"], recomputed_summary):
        raise ValueError("artifact summary does not recompute from retained run records")
    if payload["artifact_sha256"] != _digest_excluding(payload, "artifact_sha256"):
        raise ValueError("artifact content digest mismatch")
    return {
        "valid": True,
        "schema": REFERENCE_LIFE_SCORECARD_ARTIFACT_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "status": recomputed_summary["status"],
        "permanently_nonpromoting": True,
    }


def _space_descriptor(spec: SpaceSpec) -> dict[str, Any]:
    return {
        "kind": spec.kind,
        "shape": list(spec.shape),
        "dtype": spec.dtype,
        "semantic_id": spec.semantic_id,
        "cardinality": spec.cardinality,
        "low": None if spec.low is None else list(spec.low),
        "high": None if spec.high is None else list(spec.high),
    }


def _agent_manifest_descriptor(manifest: AgentManifest) -> dict[str, Any]:
    return {
        "api_version": REFERENCE_AGENT_API_VERSION,
        "schema": manifest.schema,
        "implementation_id": manifest.implementation_id,
        "state_schema": manifest.state_schema,
        "config_sha256": manifest.config_sha256,
        "manifest_id": manifest.manifest_id,
        "config": manifest.config,
        "observation_spec": _space_descriptor(manifest.observation_spec),
        "action_spec": _space_descriptor(manifest.action_spec),
        "capabilities": {
            "dispatch_rebinding": manifest.capabilities.dispatch_rebinding,
        },
    }


def _switching_environment_config(plan: ReferenceLifeDevelopmentPlan) -> SwitchingTwoStateConfig:
    protocol = plan.protocol("switching_two_state")
    return SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=protocol["phase_length"],
        payoffs_a=tuple(tuple(row) for row in protocol["payoffs_a"]),
        payoffs_b=tuple(tuple(row) for row in protocol["payoffs_b"]),
    )


def _river_environment_config(plan: ReferenceLifeDevelopmentPlan) -> RiverSwimConfig:
    protocol = plan.protocol("riverswim")
    return RiverSwimConfig(  # type: ignore[call-arg]
        n_states=protocol["n_states"],
        p_right_up=protocol["p_right_up"],
        p_right_down=protocol["p_right_down"],
        reward_left=protocol["reward_left"],
        reward_right=protocol["reward_right"],
        initial_state=protocol["initial_state"],
    )


def _prototype_agent_config(
    plan: ReferenceLifeDevelopmentPlan,
    *,
    environment_kind: str,
    arm: str,
) -> PrototypeAgentConfig:
    definition = plan.arm_definition(arm)
    config = definition["config"]
    observation_dim = (
        2
        if environment_kind == "switching_two_state"
        else cast(int, plan.protocol("riverswim")["n_states"])
    )
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(),
                observation_dim=observation_dim,
                n_primitive_actions=2,
                base_step_size=config["base_step_size"],
                base_avg_reward_step_size=config["base_average_reward_step_size"],
                base_trace_decay=config["base_trace_decay"],
                epsilon_base=config["epsilon_base"],
            )
        )
    )


def _control_adapter(
    *,
    arm: str,
    arm_definition: Mapping[str, Any],
    environment_kind: str,
    switching_config: SwitchingTwoStateConfig | None,
    river_config: RiverSwimConfig | None,
    horizon: int,
) -> Any:
    # Lazy import keeps plan/validation tooling usable while the optional run
    # surface is being installed, while still using the exact named adapters.
    from alberta_framework.reference_life_controls import (
        AnalyticOracleReferenceAdapter,
        AnalyticOracleReferenceConfig,
        DifferentialSARSAReferenceAdapter,
        DifferentialSARSAReferenceConfig,
        DiscountedSARSAReferenceAdapter,
        DiscountedSARSAReferenceConfig,
        UniformRandomReferenceAdapter,
        UniformRandomReferenceConfig,
    )

    if environment_kind == "switching_two_state":
        if switching_config is None:
            raise ValueError("switching control construction lacks its environment config")
        factory_argument: SwitchingTwoStateConfig | RiverSwimConfig = switching_config
    elif environment_kind == "riverswim":
        if river_config is None:
            raise ValueError("RiverSwim control construction lacks its environment config")
        factory_argument = river_config
    else:
        raise ValueError(f"unsupported environment {environment_kind!r}")

    if arm_definition != build_development_plan().arm_definition(arm):
        raise ValueError("control arm definition is not the pinned plan-v1 definition")
    definition_config = arm_definition.get("config")
    if not isinstance(definition_config, Mapping):
        raise ValueError("control arm definition lacks its exact config")
    config = dict(definition_config)

    if arm == "random":
        if config != {"action_distribution": "uniform", "n_actions": 2}:
            raise ValueError("random control config is outside the pinned plan")
        random_config = (
            UniformRandomReferenceConfig.for_switching(factory_argument)
            if isinstance(factory_argument, SwitchingTwoStateConfig)
            else UniformRandomReferenceConfig.for_riverswim(factory_argument)
        )
        random_payload = random_config.to_config()
        if (
            random_payload["n_actions"] != config["n_actions"]
            or random_payload["action_distribution"] != config["action_distribution"]
        ):
            raise ValueError("random control did not resolve the planned semantics")
        return UniformRandomReferenceAdapter(random_config)
    if arm == "privileged_oracle":
        if not _json_exact_equal(config, {
            "finite_horizon_solver": (
                "finite_horizon_backward_dynamic_program.float64.tie_low.preview1"
            ),
            "horizon_binding": "environment_protocol_horizon",
            "privileged_environment_model": True,
            "tie_break": "lowest_action_index",
        }):
            raise ValueError("oracle control config is outside the pinned plan")
        oracle_config = (
            AnalyticOracleReferenceConfig.for_switching(
                factory_argument,
                horizon=horizon,
            )
            if isinstance(factory_argument, SwitchingTwoStateConfig)
            else AnalyticOracleReferenceConfig.for_riverswim(
                factory_argument,
                horizon=horizon,
            )
        )
        oracle_payload = oracle_config.to_config()
        if (
            oracle_payload["privileged_environment_model"]
            != config["privileged_environment_model"]
            or oracle_payload["tie_break"] != config["tie_break"]
        ):
            raise ValueError("oracle control did not resolve the planned semantics")
        return AnalyticOracleReferenceAdapter(oracle_config)
    if arm == "differential_sarsa":
        differential_config = (
            DifferentialSARSAReferenceConfig.for_switching(factory_argument, **config)
            if isinstance(factory_argument, SwitchingTwoStateConfig)
            else DifferentialSARSAReferenceConfig.for_riverswim(
                factory_argument, **config
            )
        )
        return DifferentialSARSAReferenceAdapter(differential_config)
    if arm == "sarsa":
        discounted_config = (
            DiscountedSARSAReferenceConfig.for_switching(factory_argument, **config)
            if isinstance(factory_argument, SwitchingTwoStateConfig)
            else DiscountedSARSAReferenceConfig.for_riverswim(
                factory_argument, **config
            )
        )
        return DiscountedSARSAReferenceAdapter(discounted_config)
    raise ValueError(f"arm {arm!r} is not a control adapter")


def build_scorecard_runner(
    plan: ReferenceLifeDevelopmentPlan,
    spec: ScorecardRunSpec,
) -> ReferenceLifeRunner:
    """Build but do not initialize one canonical matched life."""

    expected = iter_run_specs(plan)[spec.schedule_index]
    if spec != expected:
        raise ValueError("run spec is not at its canonical cyclic schedule position")
    protocol = plan.protocol(spec.environment_kind)
    horizon = cast(int, protocol["horizon"])
    switching_config = (
        _switching_environment_config(plan)
        if spec.environment_kind == "switching_two_state"
        else None
    )
    river_config = (
        _river_environment_config(plan) if spec.environment_kind == "riverswim" else None
    )

    if spec.arm in ("prototype", "prototype_frozen"):
        agent_config = _prototype_agent_config(
            plan,
            environment_kind=spec.environment_kind,
            arm=spec.arm,
        )
        if switching_config is not None:
            return build_prototype_switching_life(
                agent_config=agent_config,
                environment_config=switching_config,
                lifecycle_id=spec.lifecycle_id,
                seed=spec.seed,
                max_accepted_events=horizon,
            )
        assert river_config is not None
        return build_prototype_riverswim_life(
            agent_config=agent_config,
            environment_config=river_config,
            lifecycle_id=spec.lifecycle_id,
            seed=spec.seed,
            max_accepted_events=horizon,
        )

    adapter = _control_adapter(
        arm=spec.arm,
        arm_definition=plan.arm_definition(spec.arm),
        environment_kind=spec.environment_kind,
        switching_config=switching_config,
        river_config=river_config,
        horizon=horizon,
    )
    environment: Any
    if switching_config is not None:
        dispatch_config = ExactDispatchConfig()
        metrics_config = ReferenceLifeMetricsConfig(mode="switching_two_phase")
        environment = SwitchingTwoStateReferenceEnvironment(
            switching_config,
            observation_spec=adapter.manifest.observation_spec,
            action_spec=adapter.manifest.action_spec,
            executor_id=dispatch_config.executor_id,
            executor_epoch=dispatch_config.executor_epoch,
        )
    else:
        assert river_config is not None
        dispatch_config = ExactDispatchConfig(
            executor_id="asi.riverswim.executor",
            executor_epoch="asi.riverswim.executor_epoch.1",
        )
        metrics_config = ReferenceLifeMetricsConfig(mode="stationary")
        environment = RiverSwimReferenceEnvironment(
            river_config,
            observation_spec=adapter.manifest.observation_spec,
            action_spec=adapter.manifest.action_spec,
            executor_id=dispatch_config.executor_id,
            executor_epoch=dispatch_config.executor_epoch,
        )
    return ReferenceLifeRunner.create(
        agent_adapter=adapter,
        environment_adapter=environment,
        lifecycle_id=spec.lifecycle_id,
        seed=spec.seed,
        max_accepted_events=horizon,
        dispatch_config=dispatch_config,
        metrics_config=metrics_config,
    )


def _streaming_summary_for_spec(
    plan: ReferenceLifeDevelopmentPlan,
    spec: ScorecardRunSpec,
) -> StreamingRunSummary:
    protocol = plan.protocol(spec.environment_kind)
    if spec.environment_kind == "switching_two_state":
        return StreamingRunSummary.for_switching(
            horizon=protocol["horizon"],
            phase_length=protocol["phase_length"],
            post_switch_window=protocol["post_switch_window"],
        )
    return StreamingRunSummary.for_riverswim(
        horizon=protocol["horizon"],
        early_window=protocol["early_window"],
        late_window=protocol["late_window"],
        n_states=protocol["n_states"],
    )


def _block_agent_state(state: Any) -> None:
    for leaf in _iter_array_leaves(state):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _resolved_components(
    plan: ReferenceLifeDevelopmentPlan,
    spec: ScorecardRunSpec,
    runner: ReferenceLifeRunner | None,
) -> dict[str, Any]:
    if runner is None:
        return {
            "arm_definition": plan.arm_definition(spec.arm),
            "agent_manifest": None,
            "environment_manifest": None,
            "life_config": None,
            "life_config_sha256": None,
        }
    return {
        "arm_definition": plan.arm_definition(spec.arm),
        "agent_manifest": _agent_manifest_descriptor(runner.agent_adapter.manifest),
        "environment_manifest": runner.environment_adapter.manifest.descriptor(),
        "life_config": runner.config.config,
        "life_config_sha256": runner.config.config_sha256,
    }


def _telemetry_payload(
    *,
    setup_seconds: float | None,
    cold_step_seconds: float | None,
    warmed_step_seconds_total: float,
    warmed_step_count: int,
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "policy": dict(TELEMETRY_POLICY),
        "setup_seconds": setup_seconds,
        "cold_step_seconds": cold_step_seconds,
        "warmed_step_seconds_total": warmed_step_seconds_total,
        "warmed_step_count": warmed_step_count,
        "warmed_step_seconds_mean": (
            None
            if warmed_step_count == 0
            else warmed_step_seconds_total / warmed_step_count
        ),
        "total_seconds": total_seconds,
    }


def run_scorecard_shard(
    plan: ReferenceLifeDevelopmentPlan,
    spec: ScorecardRunSpec,
) -> dict[str, Any]:
    """Execute one fresh-process shard and retain success or ordinary failure."""

    expected = iter_run_specs(plan)[spec.schedule_index]
    if spec != expected:
        raise ValueError("run spec is not at its canonical schedule position")
    source_identity, runtime_identity, dependency_identity = (
        _current_consistency_identities()
    )
    started = time.monotonic()
    stage = "build"
    runner: ReferenceLifeRunner | None = None
    state: Any | None = None
    streaming = _streaming_summary_for_spec(plan, spec)
    setup_seconds: float | None = None
    cold_step_seconds: float | None = None
    warmed_step_seconds_total = 0.0
    warmed_step_count = 0
    initial_resources: dict[str, Any] | None = None

    try:
        setup_started = time.monotonic()
        runner = build_scorecard_runner(plan, spec)
        stage = "init"
        state = runner.init()
        _block_agent_state(state.agent_state)
        setup_seconds = time.monotonic() - setup_started
        initial_resources = _agent_resource_payload(runner.agent_adapter, state.agent_state)
        stage = "step"
        while state.phase is LifePhase.QUIESCENT:
            step_started = time.monotonic()
            step = runner.step(state)
            _block_agent_state(step.state.agent_state)
            step_seconds = time.monotonic() - step_started
            if cold_step_seconds is None:
                cold_step_seconds = step_seconds
            else:
                warmed_step_seconds_total += step_seconds
                warmed_step_count += 1
            if step.accepted:
                if step.event is None:
                    raise RuntimeError("accepted runner step did not expose its event")
                observation = step.event.transaction.next_decision_observation
                if observation is None:
                    raise RuntimeError("continuing scorecard event lacks its next observation")
                observation_array = observation.to_numpy()
                if observation_array.ndim != 1:
                    raise RuntimeError("scorecard observation is not a one-hot vector")
                next_state_index = int(np.argmax(observation_array))
                streaming.observe(
                    reward=step.event.transaction.reward,
                    oracle_reward=step.event.oracle_reward,
                    regime_id=step.event.regime_id,
                    parameters_changed=step.event.step_result.parameters_changed,
                    next_state_index=next_state_index,
                )
            state = step.state
            if not step.accepted:
                reason = step.rejection_reason or "reference life rejected a step"
                raise RuntimeError(reason)
        if state.phase is not LifePhase.COMPLETED:
            reason = (
                "reference life left quiescent execution without reaching completed phase"
                if state.halt is None
                else state.halt.reason
            )
            raise RuntimeError(reason)
        outcome = streaming.finalize()
        if state.accepted_events != outcome["accepted_events"]:
            raise RuntimeError("runner and streaming accepted-event counts disagree")
        if state.environment_rng_cursor != state.accepted_events:
            raise RuntimeError("runner environment cursor is not matched to accepted events")
        if initial_resources is None:
            raise RuntimeError("initial resource accounting is unavailable")
        final_resources = _agent_resource_payload(runner.agent_adapter, state.agent_state)
        outcome.update(
            {
                "environment_rng_cursor": state.environment_rng_cursor,
                "transcript_sha256": state.transcript_sha256,
                "resources": {
                    "initial": initial_resources,
                    "final": final_resources,
                },
                "parameter_change_check": parameter_change_check(
                    spec.arm, state.metrics.parameter_change_events
                ),
            }
        )
        record = {
            "schema": REFERENCE_LIFE_SCORECARD_RUN_SCHEMA,
            "plan_sha256": plan.plan_sha256,
            "schedule_index": spec.schedule_index,
            "environment_kind": spec.environment_kind,
            "arm": spec.arm,
            "seed": spec.seed,
            "lifecycle_id": spec.lifecycle_id,
            "evidence_policy": dict(NONPROMOTING_POLICY),
            "source_identity": source_identity,
            "runtime_identity": runtime_identity,
            "dependency_identity": dependency_identity,
            "status": "completed",
            "failure": None,
            "resolved": _resolved_components(plan, spec, runner),
            "outcome": outcome,
            "partial_outcome": None,
            "telemetry": _telemetry_payload(
                setup_seconds=setup_seconds,
                cold_step_seconds=cold_step_seconds,
                warmed_step_seconds_total=warmed_step_seconds_total,
                warmed_step_count=warmed_step_count,
                total_seconds=time.monotonic() - started,
            ),
        }
        return _record_with_digest(record)
    except Exception as exc:
        accepted_events = (
            streaming.accepted_events
            if state is None
            else int(getattr(state, "accepted_events", streaming.accepted_events))
        )
        message = str(exc).strip() or repr(exc)
        failure_record = {
            "schema": REFERENCE_LIFE_SCORECARD_RUN_SCHEMA,
            "plan_sha256": plan.plan_sha256,
            "schedule_index": spec.schedule_index,
            "environment_kind": spec.environment_kind,
            "arm": spec.arm,
            "seed": spec.seed,
            "lifecycle_id": spec.lifecycle_id,
            "evidence_policy": dict(NONPROMOTING_POLICY),
            "source_identity": source_identity,
            "runtime_identity": runtime_identity,
            "dependency_identity": dependency_identity,
            "status": "failed",
            "failure": {
                "stage": stage,
                "type": type(exc).__qualname__,
                "message": message,
                "accepted_events": accepted_events,
            },
            "resolved": _resolved_components(plan, spec, runner),
            "outcome": None,
            "partial_outcome": streaming.partial(),
            "telemetry": _telemetry_payload(
                setup_seconds=setup_seconds,
                cold_step_seconds=cold_step_seconds,
                warmed_step_seconds_total=warmed_step_seconds_total,
                warmed_step_count=warmed_step_count,
                total_seconds=time.monotonic() - started,
            ),
        }
        return _record_with_digest(failure_record)


def _spec_for_identity(
    plan: ReferenceLifeDevelopmentPlan,
    *,
    environment_kind: str,
    arm: str,
    seed: int,
) -> ScorecardRunSpec:
    matches = [
        spec
        for spec in iter_run_specs(plan)
        if spec.environment_kind == environment_kind and spec.arm == arm and spec.seed == seed
    ]
    if len(matches) != 1:
        raise ValueError("requested run identity is outside the fixed plan")
    return matches[0]


def validate_scorecard_run_record(
    payload: Mapping[str, Any],
    *,
    plan: ReferenceLifeDevelopmentPlan | None = None,
    _consistency_identities: (
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Validate one shard against its canonical schedule identity."""

    effective_plan = build_development_plan() if plan is None else plan
    environment = payload.get("environment_kind")
    arm = payload.get("arm")
    seed = payload.get("seed")
    if type(environment) is not str or type(arm) is not str or type(seed) is not int:
        raise ValueError("run shard identity is malformed")
    spec = _spec_for_identity(
        effective_plan,
        environment_kind=environment,
        arm=arm,
        seed=seed,
    )
    identities = (
        _current_consistency_identities()
        if _consistency_identities is None
        else _consistency_identities
    )
    _validate_run_record(
        effective_plan,
        payload,
        spec,
        path="$",
        consistency_identities=identities,
    )
    return {
        "valid": True,
        "schema": REFERENCE_LIFE_SCORECARD_RUN_SCHEMA,
        "plan_sha256": effective_plan.plan_sha256,
        "schedule_index": spec.schedule_index,
        "status": payload["status"],
        "permanently_nonpromoting": True,
    }


def summarize_shard_files(
    paths: Sequence[Path],
    *,
    plan: ReferenceLifeDevelopmentPlan | None = None,
) -> dict[str, Any]:
    """Strictly load all 144 fresh-process shards and build the aggregate."""

    effective_plan = build_development_plan() if plan is None else plan
    expected_count = len(iter_run_specs(effective_plan))
    if len(paths) != expected_count:
        raise ValueError(f"aggregate requires exactly {expected_count} shard paths")
    records: list[dict[str, Any]] = []
    seen_files: set[tuple[int, int]] = set()
    total_bytes = 0
    consistency_identities = _current_consistency_identities()
    for path in paths:
        remaining_budget = MAX_SCORECARD_AGGREGATE_INPUT_BYTES - total_bytes
        payload, metadata = _load_json_strict_with_metadata(
            Path(path),
            max_bytes=min(MAX_SCORECARD_JSON_INPUT_BYTES, remaining_budget),
        )
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen_files:
            raise ValueError("aggregate shard paths must name unique regular files")
        seen_files.add(identity)
        total_bytes += metadata.st_size
        validate_scorecard_run_record(
            payload,
            plan=effective_plan,
            _consistency_identities=consistency_identities,
        )
        records.append(payload)
    artifact = build_scorecard_artifact(effective_plan, records)
    validate_scorecard_artifact(artifact)
    return artifact


def _load_plan(path: Path | None) -> ReferenceLifeDevelopmentPlan:
    if path is None:
        return build_development_plan()
    return ReferenceLifeDevelopmentPlan.from_payload(load_json_strict(path))


def _print_json(value: Any) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PERMANENTLY NONPROMOTING matched reference-life development scorecard"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="write the immutable canonical plan")
    plan_parser.add_argument("--output", type=Path, required=True)

    shard_parser = subparsers.add_parser(
        "run-shard",
        help="run exactly one canonical shard in this fresh Python process",
    )
    shard_parser.add_argument("--plan", type=Path)
    shard_parser.add_argument("--environment", choices=ENVIRONMENT_ROSTER, required=True)
    shard_parser.add_argument("--arm", choices=ARM_ROSTER, required=True)
    shard_parser.add_argument("--seed", type=int, choices=SEED_ROSTER, required=True)
    shard_parser.add_argument("--output", type=Path, required=True)

    summarize_parser = subparsers.add_parser(
        "summarize", help="strictly combine all canonical fresh-process shards"
    )
    summarize_parser.add_argument("--plan", type=Path)
    summarize_parser.add_argument("--output", type=Path, required=True)
    summarize_parser.add_argument("shards", type=Path, nargs="+")

    validate_parser = subparsers.add_parser(
        "validate", help="validate one shard or one aggregate"
    )
    validate_parser.add_argument("input", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the immutable nonpromoting development workflow."""

    args = _parser().parse_args(argv)
    if args.command == "plan":
        plan = build_development_plan()
        write_new_json(args.output, plan.to_payload())
        return 0
    if args.command == "run-shard":
        # Fail before source hashing, JAX setup, or life construction.
        preflight_new_output(args.output)
        plan = _load_plan(args.plan)
        spec = _spec_for_identity(
            plan,
            environment_kind=args.environment,
            arm=args.arm,
            seed=args.seed,
        )
        record = run_scorecard_shard(plan, spec)
        validate_scorecard_run_record(record, plan=plan)
        write_new_json(args.output, record)
        return 0 if record["status"] == "completed" else 1
    if args.command == "summarize":
        preflight_new_output(args.output)
        plan = _load_plan(args.plan)
        artifact = summarize_shard_files(args.shards, plan=plan)
        write_new_json(args.output, artifact)
        return 0
    if args.command == "validate":
        payload = load_json_strict(args.input)
        if payload.get("schema") == REFERENCE_LIFE_SCORECARD_PLAN_SCHEMA:
            plan = ReferenceLifeDevelopmentPlan.from_payload(payload)
            result = {
                "valid": True,
                "schema": REFERENCE_LIFE_SCORECARD_PLAN_SCHEMA,
                "plan_sha256": plan.plan_sha256,
                "evidence_policy": NONPROMOTING_POLICY,
            }
        elif payload.get("schema") == REFERENCE_LIFE_SCORECARD_RUN_SCHEMA:
            result = validate_scorecard_run_record(payload)
        elif payload.get("schema") == REFERENCE_LIFE_SCORECARD_ARTIFACT_SCHEMA:
            result = validate_scorecard_artifact(payload)
        else:
            raise ValueError("input is neither a scorecard plan, shard, nor aggregate")
        _print_json(result)
        return 0
    raise AssertionError("argparse returned an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
