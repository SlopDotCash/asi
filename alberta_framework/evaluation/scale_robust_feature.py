"""Scale-robust exhaustive-pair feature evaluation protocol.

This module contains the executable protocol behind the v2 evidence artifact.
The protocol is deliberately narrow: it evaluates an exhaustive degree-two
pair archive on the nine-segment Alberta gauntlet with a visible, supplied
two-coordinate task cue.  It does not evaluate open-ended or general feature
discovery.

Mechanism development and the one-time v2 threshold calibration used integer
seed identifiers 8--15.  Seeds 30--45 were later found to have been exposed
by an older primary-only confirmation, so v2 does not reuse the 30--59
namespace.  Its fresh 30-seed schedule is deterministically derived from a
frozen SHA-256 namespace and recorded explicitly below.  Each identifier maps
directly to ``jax.random.key(seed_id)``; every arm and seed independently
initializes the learner with ``jax.random.key(123)``.  This fixed
initialization narrows the claim to stream-seed variation and does not
establish initialization robustness.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework._seed_validation import require_unique_jax_seeds
from alberta_framework.core.interaction_features import (
    FixedBudgetInteractionLearner,
)
from alberta_framework.evaluation._measurement_validation import nonnegative_finite_real
from alberta_framework.streams.gauntlet import (
    SEGMENT_NAMES,
    GauntletConfig,
    GauntletStream,
)

DEVELOPMENT_SEEDS = tuple(range(8, 16))
EVIDENCE_SEED_NAMESPACE = "alberta-scale-robust-v2-fresh-evidence-2026-07-30:"
EVIDENCE_SEED_DERIVATION = (
    "int.from_bytes(SHA256(namespace_ascii + ascii(index)).digest()[:4], 'big'), index=0..29"
)
EVIDENCE_SEEDS = (
    2_476_612_463,
    1_530_459_451,
    3_117_203_318,
    1_536_318_901,
    2_212_445_791,
    409_707_519,
    85_437_679,
    2_333_405_102,
    308_844_560,
    1_012_174_348,
    157_165_507,
    2_644_321_438,
    3_712_695_078,
    1_472_213_755,
    4_073_988_609,
    1_231_719_225,
    453_256_333,
    1_972_643_767,
    226_233_076,
    192_990_423,
    1_417_189_072,
    2_415_162_767,
    4_048_779_818,
    2_042_867_307,
    3_039_992_193,
    1_940_572_846,
    1_556_070_712,
    874_158_329,
    3_662_884_740,
    3_052_189_747,
)
LEARNER_INIT_SEED = 123

CONDITION_PRIMARY = "scale_robust_retained"
CONDITION_LEGACY = "legacy_retained"
CONDITION_NO_RETENTION = "scale_robust_no_retention"
CONDITION_NAMES = (
    CONDITION_PRIMARY,
    CONDITION_LEGACY,
    CONDITION_NO_RETENTION,
)

EARLY_WINDOW_STEPS = 200
TAIL_WINDOW_STEPS = 500
ASYMPTOTIC_WINDOW_STEPS = 1_500

FROZEN_STREAM_CONFIGURATION: dict[str, int | float] = {
    "relevant_dim": 8,
    "irrelevant_dim": 4,
    "segment_length": 3_000,
    "noise_std": 0.1,
    "feature_std": 1.0,
    "scale_factor": 10.0,
    "drift_rate": 0.01,
    "context_noise_std": 0.05,
}

_COMMON_LEARNER_CONFIGURATION: dict[str, object] = {
    "n_features": 24,
    "n_tasks": 1,
    "step_size_output": 0.03,
    "utility_decay": 0.995,
    "replacement_interval": 100,
    "min_feature_age": 50,
    "candidate_count": 91,
    "candidate_min_age": 25,
    "promotion_margin": 1.05,
    "promotion_blend": 1.0,
    "generator_mix": [1.0, 0.0, 0.0],
    "candidate_strategy": "all_pairs",
    "utility_aggregation": "mean",
    "utility_top_k": 1,
    "utility_task_balancing": "none",
    "task_activity_decay": 0.995,
    "future_utility_mix": 0.0,
    "candidate_utility_retention_decay": None,
    "refresh_candidates": False,
    "refresh_promoted_candidate": False,
    "include_squares": False,
    "obgd_kappa": 2.0,
    "scale_normalizer_decay": 0.99,
    "scale_normalizer_epsilon": 1e-6,
}


def _condition_configuration(
    *,
    retention: float | None,
    scale_robust: bool,
    use_obgd: bool,
) -> dict[str, object]:
    configuration = dict(_COMMON_LEARNER_CONFIGURATION)
    configuration.update(
        {
            "utility_retention_decay": retention,
            "use_obgd": use_obgd,
            "scale_robust": scale_robust,
        }
    )
    return configuration


FROZEN_CONDITION_CONFIGURATIONS: dict[str, dict[str, object]] = {
    CONDITION_PRIMARY: _condition_configuration(
        retention=0.9999,
        scale_robust=True,
        use_obgd=False,
    ),
    CONDITION_LEGACY: _condition_configuration(
        retention=0.9999,
        scale_robust=False,
        use_obgd=True,
    ),
    CONDITION_NO_RETENTION: _condition_configuration(
        retention=None,
        scale_robust=True,
        use_obgd=False,
    ),
}

EXPECTED_MEMORY: dict[str, dict[str, int | bool]] = {
    CONDITION_PRIMARY: {
        "active_pair_slots": 24,
        "candidate_archive_slots": 91,
        "candidate_archive_is_counted": True,
        "active_pair_descriptor_scalars": 48,
        "candidate_pair_descriptor_scalars": 182,
        "active_output_weight_scalars": 24,
        "candidate_output_weight_scalars": 91,
        "normalizer_scalars": 116,
        "normalizer_bytes": 464,
        "persistent_array_bytes": 4_164,
    },
    CONDITION_LEGACY: {
        "active_pair_slots": 24,
        "candidate_archive_slots": 91,
        "candidate_archive_is_counted": True,
        "active_pair_descriptor_scalars": 48,
        "candidate_pair_descriptor_scalars": 182,
        "active_output_weight_scalars": 24,
        "candidate_output_weight_scalars": 91,
        "normalizer_scalars": 0,
        "normalizer_bytes": 0,
        "persistent_array_bytes": 3_700,
    },
    CONDITION_NO_RETENTION: {
        "active_pair_slots": 24,
        "candidate_archive_slots": 91,
        "candidate_archive_is_counted": True,
        "active_pair_descriptor_scalars": 48,
        "candidate_pair_descriptor_scalars": 182,
        "active_output_weight_scalars": 24,
        "candidate_output_weight_scalars": 91,
        "normalizer_scalars": 116,
        "normalizer_bytes": 464,
        "persistent_array_bytes": 4_164,
    },
}


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_finite_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return nonnegative_finite_real(name, value)


def _finite_float(value: object, *, name: str) -> float:
    return nonnegative_finite_real(name, value)


def _integer_pairs(
    value: object, *, name: str
) -> tuple[tuple[int, int], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    pairs: list[tuple[int, int]] = []
    for index, pair in enumerate(value):
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError(f"{name}[{index}] must be an exact integer pair")
        left, right = pair
        if type(left) is not int or type(right) is not int:
            raise ValueError(f"{name}[{index}] must contain integers")
        pairs.append((left, right))
    return tuple(pairs)


@dataclass(frozen=True)
class PhaseWindowRecord:
    """Sufficient prequential-error statistics for one 3,000-step phase."""

    phase_index: int
    phase_name: str
    step_count: int
    nonfinite_steps: int
    phase_squared_error_sum: float | None
    early_squared_error_sum: float | None
    early_count: int
    tail_squared_error_sum: float | None
    tail_count: int
    asymptotic_squared_error_sum: float | None
    asymptotic_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_index",
            _nonnegative_int(self.phase_index, name="phase_index"),
        )
        if type(self.phase_name) is not str:
            raise ValueError("phase_name must be a string")
        object.__setattr__(
            self,
            "step_count",
            _nonnegative_int(self.step_count, name="step_count"),
        )
        object.__setattr__(
            self,
            "nonfinite_steps",
            _nonnegative_int(self.nonfinite_steps, name="nonfinite_steps"),
        )
        object.__setattr__(
            self,
            "phase_squared_error_sum",
            _optional_finite_float(
                self.phase_squared_error_sum,
                name="phase_squared_error_sum",
            ),
        )
        object.__setattr__(
            self,
            "early_squared_error_sum",
            _optional_finite_float(
                self.early_squared_error_sum,
                name="early_squared_error_sum",
            ),
        )
        object.__setattr__(
            self,
            "early_count",
            _nonnegative_int(self.early_count, name="early_count"),
        )
        object.__setattr__(
            self,
            "tail_squared_error_sum",
            _optional_finite_float(
                self.tail_squared_error_sum,
                name="tail_squared_error_sum",
            ),
        )
        object.__setattr__(
            self,
            "tail_count",
            _nonnegative_int(self.tail_count, name="tail_count"),
        )
        object.__setattr__(
            self,
            "asymptotic_squared_error_sum",
            _optional_finite_float(
                self.asymptotic_squared_error_sum,
                name="asymptotic_squared_error_sum",
            ),
        )
        object.__setattr__(
            self,
            "asymptotic_count",
            _nonnegative_int(self.asymptotic_count, name="asymptotic_count"),
        )


@dataclass(frozen=True)
class ConditionSeedRecord:
    """Primitive record for one paired seed and one learner condition."""

    seed: int
    condition: str
    phases: tuple[PhaseWindowRecord, ...]
    end_segment_5_active_pairs: tuple[tuple[int, int], ...]
    end_segment_7_active_pairs: tuple[tuple[int, int], ...]
    final_active_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, name="seed"))
        if type(self.condition) is not str:
            raise ValueError("condition must be a string")
        if type(self.phases) is not tuple:
            raise TypeError("phases must be an exact tuple")
        if any(type(phase) is not PhaseWindowRecord for phase in self.phases):
            raise TypeError("phases must contain PhaseWindowRecord values")
        object.__setattr__(
            self,
            "end_segment_5_active_pairs",
            _integer_pairs(
                self.end_segment_5_active_pairs,
                name="end_segment_5_active_pairs",
            ),
        )
        object.__setattr__(
            self,
            "end_segment_7_active_pairs",
            _integer_pairs(
                self.end_segment_7_active_pairs,
                name="end_segment_7_active_pairs",
            ),
        )
        object.__setattr__(
            self,
            "final_active_pairs",
            _integer_pairs(self.final_active_pairs, name="final_active_pairs"),
        )


@dataclass(frozen=True)
class ScaleRobustFeatureReport:
    """Complete deterministic results plus host-dependent run timing."""

    seeds: tuple[int, ...]
    records: tuple[ConditionSeedRecord, ...]
    memory_by_condition: Mapping[str, Mapping[str, int | bool]]
    wall_time_seconds_by_condition: Mapping[str, float]

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple:
            raise TypeError("seeds must be an exact tuple")
        object.__setattr__(
            self,
            "seeds",
            tuple(
                _nonnegative_int(seed, name=f"seeds[{index}]")
                for index, seed in enumerate(self.seeds)
            ),
        )
        if type(self.records) is not tuple:
            raise TypeError("records must be an exact tuple")
        if any(type(record) is not ConditionSeedRecord for record in self.records):
            raise TypeError("records must contain ConditionSeedRecord values")
        if not isinstance(self.wall_time_seconds_by_condition, Mapping):
            raise TypeError("wall_time_seconds_by_condition must be a mapping")
        object.__setattr__(
            self,
            "wall_time_seconds_by_condition",
            {
                name: _finite_float(
                    value,
                    name=f"wall_time_seconds_by_condition[{name}]",
                )
                for name, value in self.wall_time_seconds_by_condition.items()
            },
        )


def frozen_stream_config() -> GauntletConfig:
    """Construct the explicit v2 stream and detect upstream default drift."""

    defaults = dataclasses.asdict(GauntletConfig())
    if defaults != FROZEN_STREAM_CONFIGURATION:
        raise RuntimeError("GauntletConfig defaults no longer match the frozen v2 protocol")
    return GauntletConfig(
        relevant_dim=8,
        irrelevant_dim=4,
        segment_length=3_000,
        noise_std=0.1,
        feature_std=1.0,
        scale_factor=10.0,
        drift_rate=0.01,
        context_noise_std=0.05,
    )


def make_condition_learner(condition: str) -> FixedBudgetInteractionLearner:
    """Construct one exact frozen learner arm."""

    if condition not in FROZEN_CONDITION_CONFIGURATIONS:
        raise ValueError(f"unknown condition: {condition!r}")
    retention = None if condition == CONDITION_NO_RETENTION else 0.9999
    scale_robust = condition != CONDITION_LEGACY
    use_obgd = condition == CONDITION_LEGACY
    learner = FixedBudgetInteractionLearner(
        n_features=24,
        n_tasks=1,
        step_size_output=0.03,
        utility_decay=0.995,
        replacement_interval=100,
        min_feature_age=50,
        candidate_count=91,
        candidate_min_age=25,
        promotion_margin=1.05,
        promotion_blend=1.0,
        generator_mix=(1.0, 0.0, 0.0),
        candidate_strategy="all_pairs",
        utility_aggregation="mean",
        utility_top_k=1,
        utility_task_balancing="none",
        task_activity_decay=0.995,
        future_utility_mix=0.0,
        utility_retention_decay=retention,
        candidate_utility_retention_decay=None,
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        include_squares=False,
        use_obgd=use_obgd,
        obgd_kappa=2.0,
        scale_robust=scale_robust,
        scale_normalizer_decay=0.99,
        scale_normalizer_epsilon=1e-6,
    )
    observed = learner.to_config()
    expected = {
        "type": "FixedBudgetInteractionLearner",
        **FROZEN_CONDITION_CONFIGURATIONS[condition],
    }
    if observed != expected:
        raise RuntimeError(f"{condition} constructor no longer matches its frozen v2 configuration")
    return learner


def count_relevant_context_pairs(
    pairs: Sequence[tuple[int, int]],
    *,
    relevant_dim: int = 8,
    input_dim: int = 12,
) -> int:
    """Count unique canonical context products over relevant input channels."""

    relevant_pairs = {
        (min(left, right), max(left, right))
        for left, right in pairs
        if ((left >= input_dim) ^ (right >= input_dim)) and min(left, right) < relevant_dim
    }
    return len(relevant_pairs)


def count_relevant_context_pairs_by_task(
    pairs: Sequence[tuple[int, int]],
    *,
    relevant_dim: int = 8,
    input_dim: int = 12,
) -> tuple[int, int]:
    """Count unique relevant products for the supplied C and D cues separately."""

    canonical_pairs = {(min(left, right), max(left, right)) for left, right in pairs}
    counts = tuple(
        sum(
            1
            for left, right in canonical_pairs
            if left < relevant_dim and right == input_dim + context_offset
        )
        for context_offset in range(2)
    )
    return counts[0], counts[1]


def _finite_window_sum(values: NDArray[np.floating[Any]]) -> float | None:
    if not bool(np.all(np.isfinite(values))):
        return None
    total = float(np.sum(values, dtype=np.float64))
    return total if math.isfinite(total) else None


def phase_window_records(
    squared_errors: NDArray[np.floating[Any]],
    config: GauntletConfig,
) -> tuple[PhaseWindowRecord, ...]:
    """Reduce one per-step trace to the exact windows used by the v2 gate."""

    errors = np.asarray(squared_errors)
    if config.segment_length < ASYMPTOTIC_WINDOW_STEPS:
        raise ValueError("segment_length must cover the frozen 1,500-step asymptotic window")
    if errors.shape != (config.num_steps,):
        raise ValueError(
            f"squared_errors must have shape {(config.num_steps,)}, got {errors.shape}"
        )
    records: list[PhaseWindowRecord] = []
    for phase_index, phase_name in enumerate(SEGMENT_NAMES):
        start = phase_index * config.segment_length
        phase = errors[start : start + config.segment_length]
        early = phase[:EARLY_WINDOW_STEPS]
        tail = phase[-TAIL_WINDOW_STEPS:]
        asymptotic = phase[-ASYMPTOTIC_WINDOW_STEPS:]
        records.append(
            PhaseWindowRecord(
                phase_index=phase_index,
                phase_name=phase_name,
                step_count=config.segment_length,
                nonfinite_steps=int(np.count_nonzero(~np.isfinite(phase))),
                phase_squared_error_sum=_finite_window_sum(phase),
                early_squared_error_sum=_finite_window_sum(early),
                early_count=EARLY_WINDOW_STEPS,
                tail_squared_error_sum=_finite_window_sum(tail),
                tail_count=TAIL_WINDOW_STEPS,
                asymptotic_squared_error_sum=_finite_window_sum(asymptotic),
                asymptotic_count=ASYMPTOTIC_WINDOW_STEPS,
            )
        )
    return tuple(records)


def _canonical_pairs(
    left: NDArray[np.integer[Any]],
    right: NDArray[np.integer[Any]],
) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        (int(left_value), int(right_value))
        for left_value, right_value in zip(left.tolist(), right.tolist(), strict=True)
    )
    return pairs


def _run_condition(
    condition: str,
    seeds: tuple[int, ...],
    stream: GauntletStream,
) -> tuple[
    tuple[ConditionSeedRecord, ...],
    dict[str, int | bool],
    float,
]:
    learner = make_condition_learner(condition)
    initial_state = learner.init(stream.feature_dim, jr.key(LEARNER_INIT_SEED))
    memory = learner.memory_accounting(initial_state)
    keys = jnp.stack([jr.key(seed) for seed in seeds])
    end_segment_5_step = 6 * stream.config.segment_length - 1
    end_segment_7_step = 8 * stream.config.segment_length - 1

    def one_seed(
        key: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        learner_state = learner.init(
            stream.feature_dim,
            jr.key(LEARNER_INIT_SEED),
        )
        stream_state = stream.init(key)
        end_segment_5_left = jnp.zeros_like(learner_state.feature_left)
        end_segment_5_right = jnp.zeros_like(learner_state.feature_right)
        end_segment_7_left = jnp.zeros_like(learner_state.feature_left)
        end_segment_7_right = jnp.zeros_like(learner_state.feature_right)

        def step_fn(
            carry: tuple[Any, Any, Array, Array, Array, Array],
            index: Array,
        ) -> tuple[tuple[Any, Any, Array, Array, Array, Array], Array]:
            (
                state,
                gauntlet_state,
                saved_end_5_left,
                saved_end_5_right,
                saved_end_7_left,
                saved_end_7_right,
            ) = carry
            timestep, next_gauntlet_state = stream.step(gauntlet_state, index)
            result = learner.update(
                state,
                timestep.observation,
                timestep.target,
            )
            capture_end_5 = index == end_segment_5_step
            saved_end_5_left = jnp.where(
                capture_end_5,
                result.state.feature_left,
                saved_end_5_left,
            )
            saved_end_5_right = jnp.where(
                capture_end_5,
                result.state.feature_right,
                saved_end_5_right,
            )
            capture_end_7 = index == end_segment_7_step
            saved_end_7_left = jnp.where(
                capture_end_7,
                result.state.feature_left,
                saved_end_7_left,
            )
            saved_end_7_right = jnp.where(
                capture_end_7,
                result.state.feature_right,
                saved_end_7_right,
            )
            squared_error = jnp.squeeze(result.errors) ** 2
            return (
                result.state,
                next_gauntlet_state,
                saved_end_5_left,
                saved_end_5_right,
                saved_end_7_left,
                saved_end_7_right,
            ), squared_error

        (
            (
                final_state,
                _,
                saved_end_5_left,
                saved_end_5_right,
                saved_end_7_left,
                saved_end_7_right,
            ),
            squared_errors,
        ) = jax.lax.scan(
            step_fn,
            (
                learner_state,
                stream_state,
                end_segment_5_left,
                end_segment_5_right,
                end_segment_7_left,
                end_segment_7_right,
            ),
            jnp.arange(stream.config.num_steps),
        )
        return (
            squared_errors,
            saved_end_5_left,
            saved_end_5_right,
            saved_end_7_left,
            saved_end_7_right,
            final_state.feature_left,
            final_state.feature_right,
        )

    started = time.perf_counter()
    (
        squared_errors,
        end_5_left,
        end_5_right,
        end_7_left,
        end_7_right,
        final_left,
        final_right,
    ) = jax.vmap(one_seed)(keys)
    (
        squared_errors,
        end_5_left,
        end_5_right,
        end_7_left,
        end_7_right,
        final_left,
        final_right,
    ) = jax.device_get(
        (
            squared_errors,
            end_5_left,
            end_5_right,
            end_7_left,
            end_7_right,
            final_left,
            final_right,
        )
    )
    elapsed = time.perf_counter() - started

    squared_errors_np = np.asarray(squared_errors)
    end_5_left_np = np.asarray(end_5_left)
    end_5_right_np = np.asarray(end_5_right)
    end_7_left_np = np.asarray(end_7_left)
    end_7_right_np = np.asarray(end_7_right)
    final_left_np = np.asarray(final_left)
    final_right_np = np.asarray(final_right)
    records: list[ConditionSeedRecord] = []
    for index, seed in enumerate(seeds):
        records.append(
            ConditionSeedRecord(
                seed=seed,
                condition=condition,
                phases=phase_window_records(
                    squared_errors_np[index],
                    stream.config,
                ),
                end_segment_5_active_pairs=_canonical_pairs(
                    end_5_left_np[index],
                    end_5_right_np[index],
                ),
                end_segment_7_active_pairs=_canonical_pairs(
                    end_7_left_np[index],
                    end_7_right_np[index],
                ),
                final_active_pairs=_canonical_pairs(
                    final_left_np[index],
                    final_right_np[index],
                ),
            )
        )
    return tuple(records), memory, elapsed


def run_scale_robust_feature_evaluation(
    *,
    seeds: Sequence[int] = EVIDENCE_SEEDS,
) -> ScaleRobustFeatureReport:
    """Run the three paired frozen arms in one uninterrupted life per seed.

    The public default is the fresh namespace-derived 30-seed schedule.
    Alternate schedules are useful only for development diagnostics; the v2
    evidence validator will not accept them.

    Raises:
        ValueError: If ``seeds`` is empty, contains a non-canonical-``int``
            value (a ``bool``, a ``float`` that would otherwise be silently
            truncated by ``int()``, a numpy/JAX scalar, or a numeric
            string), a duplicate, or a value outside the JAX uint32 scalar
            key domain ``[0, 2**32 - 1]`` -- ``jax.random.key`` reduces an
            out-of-range seed modulo ``2**32``, so two distinct declared
            seeds could otherwise silently execute the identical random
            stream while being recorded as different identities.
    """

    if type(seeds) not in (list, tuple):
        raise ValueError("seeds must be an actual list or tuple")
    seed_schedule = require_unique_jax_seeds(seeds, name="seeds")

    stream = GauntletStream(frozen_stream_config())
    records: list[ConditionSeedRecord] = []
    memory_by_condition: dict[str, Mapping[str, int | bool]] = {}
    wall_times: dict[str, float] = {}
    for condition in CONDITION_NAMES:
        condition_records, memory, elapsed = _run_condition(
            condition,
            seed_schedule,
            stream,
        )
        records.extend(condition_records)
        memory_by_condition[condition] = memory
        wall_times[condition] = elapsed
    return ScaleRobustFeatureReport(
        seeds=seed_schedule,
        records=tuple(records),
        memory_by_condition=memory_by_condition,
        wall_time_seconds_by_condition=wall_times,
    )


def frozen_configuration_payload() -> dict[str, object]:
    """Return a fresh JSON-compatible copy of the complete v2 configuration."""

    return {
        "stream": dict(FROZEN_STREAM_CONFIGURATION),
        "seed_key_derivation": "jax.random.key(seed_id)",
        "evidence_seed_namespace": EVIDENCE_SEED_NAMESPACE,
        "evidence_seed_derivation": EVIDENCE_SEED_DERIVATION,
        "learner_init_seed": LEARNER_INIT_SEED,
        "windows": {
            "full_phase_steps": FROZEN_STREAM_CONFIGURATION["segment_length"],
            "early_steps": EARLY_WINDOW_STEPS,
            "tail_steps": TAIL_WINDOW_STEPS,
            "asymptotic_steps": ASYMPTOTIC_WINDOW_STEPS,
        },
        "conditions": {
            name: dict(FROZEN_CONDITION_CONFIGURATIONS[name]) for name in CONDITION_NAMES
        },
    }


def validate_runtime_memory(
    memory_by_condition: Mapping[str, Mapping[str, int | bool]],
) -> bool:
    """Return whether runtime accounting matches the frozen counted budget."""

    return {
        condition: dict(memory_by_condition.get(condition, {})) for condition in CONDITION_NAMES
    } == EXPECTED_MEMORY


__all__ = [
    "ASYMPTOTIC_WINDOW_STEPS",
    "CONDITION_LEGACY",
    "CONDITION_NAMES",
    "CONDITION_NO_RETENTION",
    "CONDITION_PRIMARY",
    "ConditionSeedRecord",
    "DEVELOPMENT_SEEDS",
    "EARLY_WINDOW_STEPS",
    "EVIDENCE_SEED_DERIVATION",
    "EVIDENCE_SEED_NAMESPACE",
    "EVIDENCE_SEEDS",
    "EXPECTED_MEMORY",
    "FROZEN_CONDITION_CONFIGURATIONS",
    "FROZEN_STREAM_CONFIGURATION",
    "LEARNER_INIT_SEED",
    "PhaseWindowRecord",
    "ScaleRobustFeatureReport",
    "TAIL_WINDOW_STEPS",
    "count_relevant_context_pairs",
    "count_relevant_context_pairs_by_task",
    "frozen_configuration_payload",
    "make_condition_learner",
    "phase_window_records",
    "run_scale_robust_feature_evaluation",
    "validate_runtime_memory",
]
