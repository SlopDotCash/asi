"""Prospective optimization-readiness diagnostic for development evaluation.

The equations and empirical estimator follow Wang et al., arXiv:2605.09044v1.
This module evaluates caller-provided measurements; it neither trains nor reads
benchmark data, and therefore cannot itself create a performance result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

PROTOCOL_SCHEMA = "asi.optimization-readiness.protocol.v1"
RESOURCE_SCHEMA = "asi.optimization-readiness.resources.v1"
RESULT_SCHEMA = "asi.optimization-readiness.development-result.v1"

_INT32_MAX: Final[int] = (1 << 31) - 1
_UINT32_MAX: Final[int] = (1 << 32) - 1
_MAX_STRING_BYTES: Final[int] = 4_096
_MAX_INFORMATION_ITEMS: Final[int] = 256
_MAX_RESULT_COUNT: Final[int] = 256
_MAX_PERSISTENT_BYTES: Final[int] = 256 * 1024 * 1024
_MAX_WORKING_SET_BYTES: Final[int] = 512 * 1024 * 1024
_MAX_ARRAY_BYTES: Final[int] = 512 * 1024 * 1024
_FLOAT64_BYTES: Final[int] = np.dtype(np.float64).itemsize
_APPENDIX_C1_VALIDATION_OBSERVATIONS: Final[int] = 10_000
_APPENDIX_C1_BATCH_SIZE: Final[int] = 4
_APPENDIX_C1_BATCH_COUNT: Final[int] = 128
_APPENDIX_C1_ROLLOUT_COUNT: Final[int] = 128
_APPENDIX_C1_GAIN_STEPS: Final[frozenset[int]] = frozenset({1, 10, 100})
_PERMITTED_BOUNDARY_INFORMATION: Final[tuple[str, ...]] = ("task_start",)
_PERMITTED_TASK_INFORMATION: Final[tuple[str, ...]] = (
    "current_validation_inputs",
    "current_validation_labels",
)
_SAMPLING_PROVENANCE: Final[str] = (
    "caller_reported_independent_with_replacement_not_verified_from_gradients"
)

OPTIMIZATION_READINESS_PROTOCOL = MappingProxyType(
    {
        "schema": PROTOCOL_SCHEMA,
        "paper_revision": "arXiv:2605.09044v1",
        "paper_revision_date": "2026-05-09",
        "population_gradient_estimator": "full_validation_set_gradient",
        "reliability_estimator": "caller-provided_minibatch_gradients",
        "reference_reliability_batch_count": 128,
        "reference_reliability_batch_size": 4,
        "official_code_revision": "none-cited-in-arxiv-v1-as-of-2026-08-17",
        "estimator": "generic-equation-helper-plus-appendix-c.1-declared-mode",
        "paper_defaults": MappingProxyType(
            {
                "validation_observations": 10_000,
                "mini_batch_size": 4,
                "batch_count": 128,
                "future_gain_rollout_count": 128,
                "future_gain_step_size": 1e-3,
            }
        ),
        "asi_protocol_differences": (
            "caller_supplies_precomputed gradients",
            "gradient arrays cannot prove sample membership or independence",
            "result metrics, outcomes, and execution identities are caller-reported",
        ),
        "diagnostics": (
            "optimization_readiness",
            "gradient_norm",
            "representation_energy_rank_0.99",
            "curvature_energy_rank_0.99",
            "parameter_norm",
        ),
        "target": "future_relative_loss_reduction_after_matched_updates",
        "future_gain_steps": (1, 10, 100),
        "primary_comparison": "pairwise_checkpoint_ranking_accuracy",
        "matched_axes": (
            "seed",
            "checkpoint",
            "task",
            "updates",
            "observations",
            "full_validation_observations",
            "mini_batch_size",
            "diagnostic_batch_count",
            "future_gain_steps",
            "future_gain_rollout_count",
            "future_gain_batch_size",
            "future_gain_step_size",
            "parameter_count",
            "sampling_provenance",
            "allowed_boundary_information",
            "allowed_task_information",
        ),
        "resource_fields": (
            "persistent_bytes",
            "peak_working_set_bytes",
            "environment_steps",
            "data_steps",
            "model_queries",
            "parameter_updates",
            "timing_seconds",
        ),
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retention_required": True,
        "execution_protocol_required": True,
        "resource_receipts_are_authenticated": False,
        "metrics_are_recomputed_by_this_module": False,
        "outcomes_are_derived_by_this_module": False,
        "completed_result_exists": False,
    }
)


@dataclass(frozen=True)
class OptimizationReadiness:
    """Equation-level diagnostic values for one checkpoint/task pair."""

    gradient_squared_norm: float
    expected_batch_gradient_squared_norm: float
    gradient_strength: float
    gradient_reliability: float
    optimization_readiness: float
    gradient_norm: float
    batch_count: int
    parameter_count: int


@dataclass(frozen=True)
class ProspectiveDevelopmentProtocol:
    """Matched axes and permitted information for one prospective comparison arm."""

    seed: int
    checkpoint: str
    task: str
    updates: int
    observations: int
    full_validation_observations: int
    mini_batch_size: int
    diagnostic_batch_count: int
    future_gain_steps: int
    future_gain_rollout_count: int
    future_gain_batch_size: int
    future_gain_step_size: float
    parameter_count: int
    sampling_provenance: str
    allowed_boundary_information: tuple[str, ...]
    allowed_task_information: tuple[str, ...]


@dataclass(frozen=True)
class ResourceReceipt:
    """Bounded caller-reported resource accounting for a prospective result."""

    persistent_bytes: int
    peak_working_set_bytes: int
    environment_steps: int
    data_steps: int
    model_queries: int
    parameter_updates: int
    timing_seconds: float


@dataclass(frozen=True)
class DevelopmentResultReceipt:
    """Schema-validated, permanently nonpromoting prospective result report."""

    comparison_id: str
    arm_id: str
    protocol: ProspectiveDevelopmentProtocol
    resources: ResourceReceipt
    optimization_readiness: float
    gradient_norm: float
    representation_energy_rank_0_99: int
    curvature_energy_rank_0_99: int
    parameter_norm: float
    future_relative_loss_reduction: float
    reported_outcome: str


def _finite_array(
    value: object, *, name: str, dimensions: int
) -> NDArray[np.float64]:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact numpy.ndarray")
    raw = value
    if raw.ndim != dimensions or any(size < 1 for size in raw.shape):
        raise ValueError(f"{name} must be a non-empty {dimensions}-dimensional array")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must have a real numeric dtype")
    if raw.nbytes > _MAX_ARRAY_BYTES or raw.size > _MAX_ARRAY_BYTES // _FLOAT64_BYTES:
        raise ValueError(f"{name} exceeds the bounded numeric payload")
    resolved = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(resolved)):
        raise ValueError(f"{name} must contain only finite values")
    return resolved


def _squared_norm(value: NDArray[np.float64], *, name: str) -> float:
    scale = float(np.max(np.abs(value)))
    if scale == 0.0:
        return 0.0
    scaled_squared_norm = float(np.sum(np.square(value / scale)))
    if scale > math.sqrt(np.finfo(np.float64).max / scaled_squared_norm):
        raise ValueError(f"{name} squared norm must fit in a finite float64")
    result = scale * scale * scaled_squared_norm
    if not math.isfinite(result):
        raise ValueError(f"{name} squared norm must fit in a finite float64")
    return result


def _mean_squared_norm(values: NDArray[np.float64], *, name: str) -> float:
    """Average finite squared norms without an overflowing intermediate sum.

    A direct ``np.mean`` accumulates the sum before dividing, so a batch of
    large-but-finite row norms reaches infinity even though their mean is
    representable.  Rescaling by a power of two is exact, keeps every summand in
    ``[0, 1]``, and therefore preserves the mean's magnitude domain.
    """
    maximum = float(np.max(values))
    if maximum == 0.0:
        return 0.0
    _, exponent = math.frexp(maximum)
    scaled = float(np.mean(np.ldexp(values, -exponent)))
    result = math.ldexp(scaled, exponent)
    if not math.isfinite(result):
        raise ValueError(f"{name} must fit in a finite float64")
    return result


def estimate_optimization_readiness(
    *,
    loss: float,
    full_validation_gradient: object,
    batch_gradients: object,
    include_reliability: bool = True,
) -> OptimizationReadiness:
    """Evaluate the OR equations for bounded caller-provided gradients.

    This generic equation helper does not know how the gradients were sampled.
    In particular, array values cannot establish mini-batch size, membership,
    replacement, or independence.  Use
    :func:`estimate_appendix_c1_optimization_readiness` when the observable
    Appendix C.1 shape and caller-reported sampling design must be checked.
    Setting ``include_reliability=False`` is the predeclared
    gradient-strength-only mechanism-off reduction.
    """
    if type(loss) is not float and type(loss) is not int:
        raise ValueError("loss must be a finite non-negative real number")
    try:
        resolved_loss = float(loss)
    except (OverflowError, ValueError) as exc:
        raise ValueError("loss must be a finite non-negative real number") from exc
    if not math.isfinite(resolved_loss) or resolved_loss < 0.0:
        raise ValueError("loss must be a finite non-negative real number")
    if type(include_reliability) is not bool:
        raise ValueError("include_reliability must be a bool")
    gradient = _finite_array(
        full_validation_gradient,
        name="full_validation_gradient",
        dimensions=1,
    )
    gradients = _finite_array(batch_gradients, name="batch_gradients", dimensions=2)
    if gradients.shape[1] != gradient.shape[0]:
        raise ValueError(
            "full_validation_gradient and batch_gradients must share a parameter axis"
        )
    gradient_squared_norm = _squared_norm(
        gradient, name="full_validation_gradient"
    )
    row_squared_norms = np.asarray(
        [_squared_norm(row, name="batch gradient") for row in gradients],
        dtype=np.float64,
    )
    expected_squared_norm = _mean_squared_norm(
        row_squared_norms, name="expected batch gradient squared norm"
    )
    strength = gradient_squared_norm / resolved_loss if resolved_loss > 0.0 else 0.0
    reliability = (
        gradient_squared_norm / expected_squared_norm if expected_squared_norm > 0.0 else 0.0
    )
    if not include_reliability:
        readiness = strength
    elif resolved_loss > 0.0 and expected_squared_norm > 0.0:
        readiness = strength * reliability
    else:
        readiness = 0.0
    outputs = (
        gradient_squared_norm,
        expected_squared_norm,
        strength,
        reliability,
        readiness,
    )
    if not all(math.isfinite(value) for value in outputs):
        raise ValueError("optimization-readiness outputs must be finite")
    return OptimizationReadiness(
        gradient_squared_norm=gradient_squared_norm,
        expected_batch_gradient_squared_norm=expected_squared_norm,
        gradient_strength=strength,
        gradient_reliability=reliability,
        optimization_readiness=readiness,
        gradient_norm=math.sqrt(gradient_squared_norm),
        batch_count=int(gradients.shape[0]),
        parameter_count=int(gradients.shape[1]),
    )


def estimate_appendix_c1_optimization_readiness(
    *,
    loss: float,
    full_validation_gradient: object,
    batch_gradients: object,
    full_validation_observations: int,
    mini_batch_size: int,
    sampling_provenance: str,
    include_reliability: bool = True,
) -> OptimizationReadiness:
    """Validate the observable Appendix C.1 contract and evaluate OR.

    The function requires the paper's 10,000-observation validation set and
    128 reported size-4 mini-batch gradients.  ``sampling_provenance`` is an
    explicit caller report: precomputed gradient arrays cannot prove that the
    underlying samples were independent or drawn with replacement.
    """
    if (
        type(full_validation_observations) is not int
        or full_validation_observations != _APPENDIX_C1_VALIDATION_OBSERVATIONS
    ):
        raise ValueError("Appendix C.1 requires exactly 10,000 validation observations")
    if type(mini_batch_size) is not int or mini_batch_size != _APPENDIX_C1_BATCH_SIZE:
        raise ValueError("Appendix C.1 requires reported mini-batch size 4")
    if type(sampling_provenance) is not str or sampling_provenance != _SAMPLING_PROVENANCE:
        raise ValueError("Appendix C.1 sampling provenance must use the explicit unverified label")
    if type(batch_gradients) is not np.ndarray:
        raise ValueError("batch_gradients must be an exact numpy.ndarray")
    if batch_gradients.ndim != 2 or batch_gradients.shape[0] != _APPENDIX_C1_BATCH_COUNT:
        raise ValueError("Appendix C.1 requires exactly 128 mini-batch gradient rows")
    return estimate_optimization_readiness(
        loss=loss,
        full_validation_gradient=full_validation_gradient,
        batch_gradients=batch_gradients,
        include_reliability=include_reliability,
    )


def energy_rank(matrix: object, *, threshold: float = 0.99) -> int:
    """Return the smallest singular-value count reaching squared-energy mass."""
    if type(threshold) is not float:
        raise ValueError("threshold must be a float in (0, 1]")
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be a float in (0, 1]")
    resolved = _finite_array(matrix, name="matrix", dimensions=2)
    try:
        singular_values = np.linalg.svd(resolved, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("matrix singular-value decomposition did not converge") from exc
    leading = float(singular_values[0])
    if leading == 0.0:
        return 0
    squared = np.square(singular_values / leading)
    total = float(np.sum(squared))
    target = np.nextafter(threshold * total, -math.inf)
    if not math.isfinite(total):
        raise ValueError("matrix singular-value energy must be finite")
    found = int(np.searchsorted(np.cumsum(squared), target, side="left") + 1)
    return min(found, int(singular_values.shape[0]))


def _strict_mapping(value: object, *, name: str, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact dict")
    actual_keys = tuple(value.keys())
    if any(type(key) is not str for key in actual_keys):
        raise ValueError(f"{name} keys must be exact strings")
    actual = set(actual_keys)
    if actual != keys:
        raise ValueError(f"{name} keys must be exactly {sorted(keys)}")
    return value


def _strict_nonnegative_int(
    value: object,
    *,
    name: str,
    positive: bool = False,
    maximum: int = _INT32_MAX,
) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= maximum:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a bounded {qualifier} built-in int")
    return value


def _strict_finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (nonnegative and value < 0.0):
        qualifier = " finite nonnegative" if nonnegative else " finite"
        raise ValueError(f"{name} must be a{qualifier} float")
    return value


def _strict_string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be an exact list")
    if len(value) > _MAX_INFORMATION_ITEMS:
        raise ValueError(f"{name} exceeds its bounded item count")
    resolved = tuple(_strict_string(item, name=f"{name} item") for item in value)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates")
    return resolved


def _strict_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if not encoded or len(encoded) > _MAX_STRING_BYTES or "\x00" in value:
        raise ValueError(f"{name} must be a bounded non-empty string")
    return value


def _checked_product(name: str, *factors: int, maximum: int = _INT32_MAX) -> int:
    product = 1
    for factor in factors:
        if factor and product > maximum // factor:
            raise ValueError(f"{name} exceeds its bounded accounting domain")
        product *= factor
    return product


def _checked_sum(name: str, *values: int, maximum: int = _INT32_MAX) -> int:
    total = 0
    for value in values:
        if total > maximum - value:
            raise ValueError(f"{name} exceeds its bounded accounting domain")
        total += value
    return total


def validate_development_result(payload: object) -> DevelopmentResultReceipt:
    """Validate a bounded, explicitly nonauthenticated development report."""
    outer = _strict_mapping(
        payload,
        name="result",
        keys={
            "schema",
            "comparison_id",
            "arm_id",
            "protocol",
            "resources",
            "metrics",
            "reported_outcome",
            "reported_outcome_retained",
            "development_only",
            "scientific_promotion_allowed",
        },
    )
    if _strict_string(outer["schema"], name="result schema") != RESULT_SCHEMA:
        raise ValueError("result schema is not supported")
    comparison_id = _strict_string(outer["comparison_id"], name="comparison_id")
    arm_id = _strict_string(outer["arm_id"], name="arm_id")
    reported_outcome = _strict_string(
        outer["reported_outcome"], name="reported_outcome"
    )
    if reported_outcome not in {
        "supported",
        "rejected",
        "inconclusive",
    }:
        raise ValueError("reported_outcome must be supported, rejected, or inconclusive")
    if outer["reported_outcome_retained"] is not True:
        raise ValueError("reported_outcome_retained must permanently remain True")
    if outer["development_only"] is not True:
        raise ValueError("development_only must permanently remain True")
    if outer["scientific_promotion_allowed"] is not False:
        raise ValueError("scientific_promotion_allowed must permanently remain False")

    protocol_payload = _strict_mapping(
        outer["protocol"],
        name="protocol",
        keys={
            "schema",
            "seed",
            "checkpoint",
            "task",
            "updates",
            "observations",
            "full_validation_observations",
            "mini_batch_size",
            "diagnostic_batch_count",
            "future_gain_steps",
            "future_gain_rollout_count",
            "future_gain_batch_size",
            "future_gain_step_size",
            "parameter_count",
            "sampling_provenance",
            "allowed_boundary_information",
            "allowed_task_information",
        },
    )
    if _strict_string(protocol_payload["schema"], name="protocol schema") != PROTOCOL_SCHEMA:
        raise ValueError("protocol schema is not supported")
    protocol = ProspectiveDevelopmentProtocol(
        seed=_strict_nonnegative_int(
            protocol_payload["seed"], name="seed", maximum=_UINT32_MAX
        ),
        checkpoint=_strict_string(protocol_payload["checkpoint"], name="checkpoint"),
        task=_strict_string(protocol_payload["task"], name="task"),
        updates=_strict_nonnegative_int(protocol_payload["updates"], name="updates", positive=True),
        observations=_strict_nonnegative_int(
            protocol_payload["observations"], name="observations", positive=True
        ),
        full_validation_observations=_strict_nonnegative_int(
            protocol_payload["full_validation_observations"],
            name="full_validation_observations",
            positive=True,
        ),
        mini_batch_size=_strict_nonnegative_int(
            protocol_payload["mini_batch_size"], name="mini_batch_size", positive=True
        ),
        diagnostic_batch_count=_strict_nonnegative_int(
            protocol_payload["diagnostic_batch_count"],
            name="diagnostic_batch_count",
            positive=True,
        ),
        future_gain_steps=_strict_nonnegative_int(
            protocol_payload["future_gain_steps"], name="future_gain_steps", positive=True
        ),
        future_gain_rollout_count=_strict_nonnegative_int(
            protocol_payload["future_gain_rollout_count"],
            name="future_gain_rollout_count",
            positive=True,
        ),
        future_gain_batch_size=_strict_nonnegative_int(
            protocol_payload["future_gain_batch_size"],
            name="future_gain_batch_size",
            positive=True,
        ),
        future_gain_step_size=_strict_finite_float(
            protocol_payload["future_gain_step_size"],
            name="future_gain_step_size",
            nonnegative=True,
        ),
        parameter_count=_strict_nonnegative_int(
            protocol_payload["parameter_count"], name="parameter_count", positive=True
        ),
        sampling_provenance=_strict_string(
            protocol_payload["sampling_provenance"], name="sampling_provenance"
        ),
        allowed_boundary_information=_strict_string_tuple(
            protocol_payload["allowed_boundary_information"],
            name="allowed_boundary_information",
        ),
        allowed_task_information=_strict_string_tuple(
            protocol_payload["allowed_task_information"], name="allowed_task_information"
        ),
    )

    resource_payload = _strict_mapping(
        outer["resources"],
        name="resources",
        keys={
            "schema",
            "persistent_bytes",
            "peak_working_set_bytes",
            "environment_steps",
            "data_steps",
            "model_queries",
            "parameter_updates",
            "timing_seconds",
            "timing_is_telemetry_only",
        },
    )
    if _strict_string(resource_payload["schema"], name="resource schema") != RESOURCE_SCHEMA:
        raise ValueError("resource schema is not supported")
    if resource_payload["timing_is_telemetry_only"] is not True:
        raise ValueError("timing_is_telemetry_only must permanently remain True")
    resources = ResourceReceipt(
        persistent_bytes=_strict_nonnegative_int(
            resource_payload["persistent_bytes"],
            name="persistent_bytes",
            maximum=_MAX_PERSISTENT_BYTES,
        ),
        peak_working_set_bytes=_strict_nonnegative_int(
            resource_payload["peak_working_set_bytes"],
            name="peak_working_set_bytes",
            maximum=_MAX_WORKING_SET_BYTES,
        ),
        environment_steps=_strict_nonnegative_int(
            resource_payload["environment_steps"], name="environment_steps"
        ),
        data_steps=_strict_nonnegative_int(resource_payload["data_steps"], name="data_steps"),
        model_queries=_strict_nonnegative_int(
            resource_payload["model_queries"], name="model_queries"
        ),
        parameter_updates=_strict_nonnegative_int(
            resource_payload["parameter_updates"], name="parameter_updates"
        ),
        timing_seconds=_strict_finite_float(
            resource_payload["timing_seconds"], name="timing_seconds", nonnegative=True
        ),
    )

    if (
        protocol.full_validation_observations != _APPENDIX_C1_VALIDATION_OBSERVATIONS
        or protocol.mini_batch_size != _APPENDIX_C1_BATCH_SIZE
        or protocol.diagnostic_batch_count != _APPENDIX_C1_BATCH_COUNT
        or protocol.future_gain_rollout_count != _APPENDIX_C1_ROLLOUT_COUNT
        or protocol.future_gain_batch_size != _APPENDIX_C1_BATCH_SIZE
        or protocol.future_gain_steps not in _APPENDIX_C1_GAIN_STEPS
        or protocol.future_gain_step_size != 1e-3
        or protocol.sampling_provenance != _SAMPLING_PROVENANCE
    ):
        raise ValueError("diagnostic and rollout sampling must match pinned Appendix C.1")
    reliability_observations = _checked_product(
        "diagnostic mini-batch observations",
        protocol.mini_batch_size,
        protocol.diagnostic_batch_count,
    )
    rollout_updates = _checked_product(
        "future-gain updates",
        protocol.future_gain_steps,
        protocol.future_gain_rollout_count,
    )
    if protocol.updates != rollout_updates:
        raise ValueError("updates must equal future-gain steps times rollout count")
    if resources.parameter_updates != protocol.updates:
        raise ValueError("parameter_updates must equal protocol updates")
    rollout_training_observations = _checked_product(
        "future-gain training observations",
        rollout_updates,
        protocol.future_gain_batch_size,
    )
    terminal_validation_observations = _checked_product(
        "future-gain terminal validation observations",
        protocol.future_gain_rollout_count,
        protocol.full_validation_observations,
    )
    # The initial full-validation loss is co-computed with the full-validation
    # gradient. Each rollout additionally charges its training mini-batches and
    # one complete terminal validation evaluation, as in Appendix C.1.
    expected_observations = _checked_sum(
        "prospective observations",
        protocol.full_validation_observations,
        reliability_observations,
        rollout_training_observations,
        terminal_validation_observations,
    )
    if protocol.observations != expected_observations:
        raise ValueError("observations must equal the pinned diagnostic and rollout charges")
    if _checked_sum(
        "environment/data steps", resources.environment_steps, resources.data_steps
    ) != protocol.observations:
        raise ValueError("environment_steps plus data_steps must equal observations")
    if resources.model_queries != protocol.observations:
        raise ValueError("model_queries must equal charged prospective observations")
    minimum_working_set_bytes = _checked_product(
        "diagnostic gradient bytes",
        protocol.diagnostic_batch_count + 1,
        protocol.parameter_count,
        _FLOAT64_BYTES,
        maximum=_MAX_WORKING_SET_BYTES,
    )
    if resources.peak_working_set_bytes < minimum_working_set_bytes:
        raise ValueError("peak_working_set_bytes is below the bound diagnostic gradients")
    if protocol.allowed_boundary_information != _PERMITTED_BOUNDARY_INFORMATION:
        raise ValueError("allowed_boundary_information exceeds the frozen protocol")
    if protocol.allowed_task_information != _PERMITTED_TASK_INFORMATION:
        raise ValueError("allowed_task_information exceeds the frozen protocol")

    metric_payload = _strict_mapping(
        outer["metrics"],
        name="metrics",
        keys={
            "optimization_readiness",
            "gradient_norm",
            "representation_energy_rank_0_99",
            "curvature_energy_rank_0_99",
            "parameter_norm",
            "future_relative_loss_reduction",
        },
    )
    representation_rank = _strict_nonnegative_int(
        metric_payload["representation_energy_rank_0_99"],
        name="representation_energy_rank_0_99",
    )
    if representation_rank > protocol.full_validation_observations:
        raise ValueError("representation_energy_rank_0_99 exceeds the validation row count")
    curvature_rank = _strict_nonnegative_int(
        metric_payload["curvature_energy_rank_0_99"],
        name="curvature_energy_rank_0_99",
    )
    if curvature_rank > protocol.parameter_count:
        raise ValueError("curvature_energy_rank_0_99 exceeds parameter_count")
    future_gain = _strict_finite_float(
        metric_payload["future_relative_loss_reduction"],
        name="future_relative_loss_reduction",
    )
    if future_gain > 1.0:
        raise ValueError("future_relative_loss_reduction cannot exceed one")
    return DevelopmentResultReceipt(
        comparison_id=comparison_id,
        arm_id=arm_id,
        protocol=protocol,
        resources=resources,
        optimization_readiness=_strict_finite_float(
            metric_payload["optimization_readiness"],
            name="optimization_readiness",
            nonnegative=True,
        ),
        gradient_norm=_strict_finite_float(
            metric_payload["gradient_norm"], name="gradient_norm", nonnegative=True
        ),
        representation_energy_rank_0_99=representation_rank,
        curvature_energy_rank_0_99=curvature_rank,
        parameter_norm=_strict_finite_float(
            metric_payload["parameter_norm"], name="parameter_norm", nonnegative=True
        ),
        future_relative_loss_reduction=future_gain,
        reported_outcome=reported_outcome,
    )


def validate_matched_development_results(
    payloads: object,
) -> tuple[DevelopmentResultReceipt, ...]:
    """Validate receipts with equal protocol axes and resource budgets.

    Timing remains telemetry and may differ.  Equality of caller-reported
    budgets does not authenticate the underlying execution.
    """
    if type(payloads) is not list and type(payloads) is not tuple:
        raise ValueError("payloads must be an exact list or tuple")
    if not 2 <= len(payloads) <= _MAX_RESULT_COUNT:
        raise ValueError("payloads must contain a bounded count of development results")
    receipts = tuple(validate_development_result(payload) for payload in payloads)
    first = receipts[0]
    if len({receipt.arm_id for receipt in receipts}) != len(receipts):
        raise ValueError("development result arm_id values must be unique")
    for receipt in receipts[1:]:
        if receipt.comparison_id != first.comparison_id:
            raise ValueError("comparison_id must match across development results")
        if receipt.protocol != first.protocol:
            raise ValueError("all predeclared matched protocol axes must be equal")
        if (
            receipt.resources.persistent_bytes,
            receipt.resources.peak_working_set_bytes,
            receipt.resources.environment_steps,
            receipt.resources.data_steps,
            receipt.resources.model_queries,
            receipt.resources.parameter_updates,
        ) != (
            first.resources.persistent_bytes,
            first.resources.peak_working_set_bytes,
            first.resources.environment_steps,
            first.resources.data_steps,
            first.resources.model_queries,
            first.resources.parameter_updates,
        ):
            raise ValueError("all caller-reported resource budgets must be equal")
    return receipts
