"""Bounded development evaluation for continual optimizer geometry controls.

This module deliberately stops before IPMNIST. It binds three paper revisions,
runs their smallest useful matrix-geometry slices on one frozen synthetic stream,
and emits a strict, permanently nonpromoting result. The FOGO and FLAD arms are
mechanism probes rather than complete optimizer ports; those differences are part
of the validated protocol rather than being left implicit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path, PosixPath
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

GEOMETRY_RESULT_SCHEMA = "asi.optimizer-geometry.streaming-matrix-result.v1"
FROZEN_GEOMETRY_CONFIG = MappingProxyType(
    {
        "seed": 20_260_817,
        "updates": 8,
        "rows": 3,
        "columns": 2,
        "newton_schulz_steps": 5,
        "muon_dual_steps": 2,
        "muon_dual_learning_rate": 0.25,
        "allowed_boundary_information": "none",
        "target_definition": "sum_of_stream_updates_projected_away_from_fixed_e0",
    }
)
GEOMETRY_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.optimizer-geometry.protocol.v2",
        "paper_revisions": (
            "arXiv:2605.08949v2",
            "arXiv:2606.10406v1",
            "arXiv:2601.07636v1",
        ),
        "stage": "frozen_small_streaming_matrix_pre_ipmnist",
        "protocol_differences": (
            "Muon-OGD is the paper-v2 NS5 and two-step dual update on one fixed constraint",
            "FOGO is only its long-term orthogonal-correction equation; no codebook, random "
            "projection, slow-fast streams, or proximal lift",
            "FLAD is only the ideal gradient-orthogonal decomposition in equation 6; no EMA "
            "approximation, Hessian-vector product, sharpness objective, or schedule",
            "the synthetic target is a mechanism diagnostic defined by the protected complement; "
            "its outcome is not comparative performance evidence",
            "resource receipts cover persistent numeric payload only; aggregate working-set and "
            "compiler/runtime temporaries are not claimed",
        ),
        "matched_axes": (
            "seed",
            "ordered_matrices",
            "updates",
            "observations",
            "allowed_boundary_information",
        ),
        "mechanism_off": "empty_basis_or_zero_gradient_exact_reduction",
        "finite_kernel_preflight_required": True,
        "persistent_numeric_bytes_accounting_required": True,
        "aggregate_working_set_bytes_claimed": False,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)

_ARM_SPECS = (
    ("muon_ns5_empty_constraints", "muon_ogd_v2_dual", "empty_constraints"),
    ("muon_ogd_v2_dual", "muon_ogd_v2_dual", "active_constraint"),
    ("fogo_empty_basis", "fogo_v1_long_term_correction", "empty_basis"),
    ("fogo_projection", "fogo_v1_long_term_correction", "active_basis"),
    ("flad_zero_gradient", "flad_v1_ideal_noise_component", "zero_gradient"),
    ("flad_ideal_noise", "flad_v1_ideal_noise_component", "active_gradient"),
)
_CONTROL_BY_CANDIDATE = {
    "muon_ogd_v2_dual": "muon_ns5_empty_constraints",
    "fogo_projection": "fogo_empty_basis",
    "flad_ideal_noise": "flad_zero_gradient",
}

_MAX_MATRIX_ELEMENTS = 1_000_000
_MAX_RESULT_BYTES = 16 * 1024 * 1024
_MAX_TEXT_BYTES = 4_096
_MAX_RESOURCE_BYTES = 256 * 1024 * 1024
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1


def _trusted_array(value: object, *, name: str) -> Array:
    actual_type = type(value)
    if not (actual_type is np.ndarray or issubclass(actual_type, (jax.Array, jax.core.Tracer))):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    result = jnp.asarray(value)
    if result.size > _MAX_MATRIX_ELEMENTS or not jnp.issubdtype(result.dtype, jnp.floating):
        raise ValueError(f"{name} must be a bounded floating array")
    return result


def orthogonal_correction_transaction(update: Array, protected_basis: Array) -> tuple[Array, Array]:
    """Return a finite orthogonal correction and caller-visible validity bit."""
    vector = _trusted_array(update, name="update")
    basis = _trusted_array(protected_basis, name="protected_basis")
    if (
        vector.ndim != 1
        or vector.size < 1
        or basis.ndim != 2
        or basis.shape[1] != vector.shape[0]
    ):
        raise ValueError("update must be a non-empty vector and basis rows must match its width")
    coordinates = basis @ vector
    projection = basis.T @ coordinates
    candidate = vector - projection
    valid = (
        jnp.all(jnp.isfinite(vector))
        & jnp.all(jnp.isfinite(basis))
        & jnp.all(jnp.isfinite(coordinates))
        & jnp.all(jnp.isfinite(projection))
        & jnp.all(jnp.isfinite(candidate))
    )
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def _unwrap_transaction(result: tuple[Array, Array], *, name: str) -> Array:
    safe, valid = result
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError(f"{name} must be finite")
    return safe


def orthogonal_correction(update: Array, protected_basis: Array) -> Array:
    """Project a vector away from row-wise orthonormal protected directions."""
    return _unwrap_transaction(
        orthogonal_correction_transaction(update, protected_basis),
        name="orthogonal correction",
    )


def _nonzero_magnitude_bits(value: Array) -> Array:
    """Report whether any entry has a magnitude bit set, subnormal entries included.

    No float comparison can answer this on a backend that flushes subnormal
    operands: ``x != 0.0`` is ``False`` for every float32 subnormal, and every
    reduction over such entries returns exactly zero, so the magnitude bit pattern
    is the only surviving witness that the input was not a zero matrix. Masking the
    sign bit keeps both signed zeros reading as zero.
    """
    width = value.dtype.itemsize
    unsigned = np.dtype(f"uint{8 * width}")
    bits = jax.lax.bitcast_convert_type(value, unsigned)
    magnitude_mask = jnp.asarray((1 << (8 * width - 1)) - 1, dtype=unsigned)
    return jnp.any(jnp.bitwise_and(bits, magnitude_mask) != 0)


def spectral_matrix_sign_transaction(matrix: Array, *, steps: int = 5) -> tuple[Array, Array]:
    """Apply Muon-OGD v2's cubic NS5 matrix-sign approximation.

    The pinned paper defines ``f(X) = 3/2 X - 1/2 XX^T X``. Frobenius
    normalization places every singular value in its convergence interval and
    preserves an exact zero for a zero matrix. A matrix whose entries the
    backend has flushed away entirely is reported invalid rather than answered
    with the zero matrix, which is the answer reserved for a zero input.
    """
    value = _trusted_array(matrix, name="matrix")
    if (
        value.ndim != 2
        or value.size == 0
        or not jnp.issubdtype(value.dtype, jnp.floating)
        or type(steps) is not int
        or steps < 1
        or steps > 32
    ):
        raise ValueError("matrix must be non-empty and steps a positive integer")
    norm = jnp.linalg.norm(value)
    # A divisor can only recover a direction the backend still holds. Subnormal
    # float32 operands are flushed, so `jnp.max(jnp.abs(value))` is exactly zero
    # for a full-rank matrix whose entries are all subnormal, and every later step
    # then agrees it is the zero matrix: the function would certify a rank-0
    # answer for a rank-2 input. Comparing that against the bitwise witness is
    # what separates the two cases, because no float comparison can: `x != 0.0`
    # and `x > 0.0` are both False for such an entry. Overflow at the other end of
    # the range is already reported invalid rather than laundered, and a magnitude
    # the arithmetic has entirely lost gets the same disposition. Entries that are
    # merely small keep a nonzero maximum and are untouched here.
    destroyed = _nonzero_magnitude_bits(value) & (jnp.max(jnp.abs(value)) == 0.0)
    valid = (
        jnp.all(jnp.isfinite(value)) & jnp.isfinite(norm) & jnp.logical_not(destroyed)
    )
    x = value / jnp.maximum(norm, jnp.asarray(1e-12, dtype=value.dtype))
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        a = x @ x.T
        next_x = 1.5 * x - 0.5 * a @ x
        valid = valid & jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(next_x))
        x = next_x
    candidate = x.T if transposed else x
    valid = valid & jnp.all(jnp.isfinite(candidate))
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def spectral_matrix_sign(matrix: Array, *, steps: int = 5) -> Array:
    """Apply Muon-OGD v2's cubic NS5 matrix-sign approximation."""
    return _unwrap_transaction(
        spectral_matrix_sign_transaction(matrix, steps=steps), name="matrix sign"
    )


def flad_noise_component_transaction(perturbation: Array, gradient: Array) -> tuple[Array, Array]:
    """Return a finite FLAD noise component and caller-visible validity bit."""
    delta = _trusted_array(perturbation, name="perturbation")
    direction = _trusted_array(gradient, name="gradient")
    if delta.shape != direction.shape or delta.ndim != 1 or delta.size < 1:
        raise ValueError("perturbation and gradient must be non-empty equal-width vectors")
    squared_norm = jnp.vdot(direction, direction).real
    numerator = jnp.vdot(direction, delta).real
    active = squared_norm > 0.0
    denominator = jnp.where(active, squared_norm, jnp.ones_like(squared_norm))
    coefficient = numerator / denominator
    projection = direction * coefficient * active.astype(delta.dtype)
    candidate = delta - projection
    valid = (
        jnp.all(jnp.isfinite(delta))
        & jnp.all(jnp.isfinite(direction))
        & jnp.isfinite(squared_norm)
        & jnp.isfinite(numerator)
        & jnp.isfinite(coefficient)
        & jnp.all(jnp.isfinite(projection))
        & jnp.all(jnp.isfinite(candidate))
    )
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def muon_ogd_dual_update_transaction(
    momentum: Array,
    constraints: Array,
    dual: Array,
    *,
    dual_learning_rate: float,
    dual_steps: int,
    newton_schulz_steps: int = 5,
) -> tuple[Array, Array, Array]:
    """Run bounded Muon-OGD and return finite values plus a validity bit."""
    value = _trusted_array(momentum, name="momentum")
    protected = _trusted_array(constraints, name="constraints")
    multipliers = _trusted_array(dual, name="dual")
    if value.ndim != 2 or value.size == 0 or not jnp.issubdtype(value.dtype, jnp.floating):
        raise ValueError("momentum must be a non-empty floating matrix")
    if protected.ndim != 3 or protected.shape[1:] != value.shape:
        raise ValueError("constraints must have shape (count, rows, columns)")
    if multipliers.ndim != 1 or multipliers.shape[0] != protected.shape[0]:
        raise ValueError("dual must contain one multiplier per constraint")
    if type(dual_learning_rate) is not float or not math.isfinite(dual_learning_rate):
        raise ValueError("dual_learning_rate must be a finite float")
    if (
        dual_learning_rate < 0.0
        or type(dual_steps) is not int
        or not 1 <= dual_steps <= 32
    ):
        raise ValueError("dual learning rate must be non-negative and dual_steps bounded positive")
    valid = (
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(protected))
        & jnp.all(jnp.isfinite(multipliers))
    )
    for _ in range(dual_steps):
        shifted = value + jnp.einsum("k,kij->ij", multipliers, protected)
        matrix_sign, sign_valid = spectral_matrix_sign_transaction(
            shifted, steps=newton_schulz_steps
        )
        conflicts = jnp.einsum("kij,ij->k", protected, matrix_sign)
        next_multipliers = multipliers - dual_learning_rate * conflicts
        valid = (
            valid
            & jnp.all(jnp.isfinite(shifted))
            & sign_valid
            & jnp.all(jnp.isfinite(conflicts))
            & jnp.all(jnp.isfinite(next_multipliers))
        )
        multipliers = next_multipliers
    shifted = value + jnp.einsum("k,kij->ij", multipliers, protected)
    update, sign_valid = spectral_matrix_sign_transaction(
        shifted, steps=newton_schulz_steps
    )
    valid = valid & jnp.all(jnp.isfinite(shifted)) & sign_valid
    safe_update = jnp.where(valid, update, jnp.zeros_like(update))
    safe_dual = jnp.where(valid, multipliers, jnp.zeros_like(multipliers))
    return safe_update, safe_dual, valid


def muon_ogd_dual_update(
    momentum: Array,
    constraints: Array,
    dual: Array,
    *,
    dual_learning_rate: float,
    dual_steps: int,
    newton_schulz_steps: int = 5,
) -> tuple[Array, Array]:
    """Compatibility wrapper that fails closed on invalid eager values."""
    update, multipliers, valid = muon_ogd_dual_update_transaction(
        momentum,
        constraints,
        dual,
        dual_learning_rate=dual_learning_rate,
        dual_steps=dual_steps,
        newton_schulz_steps=newton_schulz_steps,
    )
    if issubclass(type(valid), jax.core.Tracer):
        return (
            jnp.where(valid, update, jnp.full_like(update, jnp.nan)),
            jnp.where(valid, multipliers, jnp.full_like(multipliers, jnp.nan)),
        )
    if not bool(valid):
        raise ValueError("Muon-OGD update must be finite")
    return update, multipliers


def flad_noise_component(perturbation: Array, gradient: Array) -> Array:
    """Remove the ideal FLAD gradient-aligned perturbation component safely."""
    return _unwrap_transaction(
        flad_noise_component_transaction(perturbation, gradient),
        name="FLAD decomposition",
    )


def _frozen_stream() -> Array:
    config = FROZEN_GEOMETRY_CONFIG
    key = jr.key(cast(int, config["seed"]), impl="threefry2x32")
    shape = (
        cast(int, config["updates"]),
        cast(int, config["rows"]),
        cast(int, config["columns"]),
    )
    return jr.normal(key, shape, dtype=jnp.float32)


def _stream_sha256(stream: Array) -> str:
    canonical = np.asarray(stream, dtype="<f4")
    descriptor = f"float32:{canonical.shape}".encode()
    return hashlib.sha256(descriptor + canonical.tobytes(order="C")).hexdigest()


def _protected_geometry(rows: int, columns: int) -> tuple[Array, Array]:
    vector = jnp.zeros((rows * columns,), dtype=jnp.float32).at[0].set(1.0)
    return vector.reshape((1, rows, columns)), vector.reshape((1, rows * columns))


def _outcome(delta: float) -> str:
    if delta < -1e-7:
        return "improved"
    if delta > 1e-7:
        return "worse"
    return "tied"


def _protocol_payload() -> dict[str, object]:
    return {
        "schema": GEOMETRY_PROTOCOL["schema"],
        "paper_revisions": list(cast(tuple[str, ...], GEOMETRY_PROTOCOL["paper_revisions"])),
        "stage": GEOMETRY_PROTOCOL["stage"],
        "protocol_differences": list(
            cast(tuple[str, ...], GEOMETRY_PROTOCOL["protocol_differences"])
        ),
        "matched_axes": list(cast(tuple[str, ...], GEOMETRY_PROTOCOL["matched_axes"])),
        "mechanism_off": GEOMETRY_PROTOCOL["mechanism_off"],
        "finite_kernel_preflight_required": True,
        "persistent_numeric_bytes_accounting_required": True,
        "aggregate_working_set_bytes_claimed": False,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def _canonical_json_bytes_raw(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("geometry result must have finite canonical JSON") from error
    if len(encoded) > _MAX_RESULT_BYTES:
        raise ValueError("geometry result exceeds the byte ceiling")
    return encoded


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime_text(name: str, value: object) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"runtime {name} must be a bounded non-empty UTF-8 string")
    return value


def _runtime_identity() -> dict[str, object]:
    environment_names = (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_RANDOM_SEED_OFFSET",
        "JAX_NUM_CPU_DEVICES",
        "XLA_FLAGS",
    )
    environment: dict[str, str | None] = {}
    for name in environment_names:
        value = os.environ.get(name)
        environment[name] = None if value is None else _runtime_text(name, value)
    devices = sorted(
        (
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": _runtime_text("device platform", str(device.platform)),
                "device_kind": _runtime_text("device kind", str(device.device_kind)),
            }
            for device in jax.devices()
        ),
        key=lambda value: (
            value["process_index"],
            value["id"],
            value["platform"],
            value["device_kind"],
        ),
    )
    if not 1 <= len(devices) <= 128:
        raise ValueError("runtime device inventory exceeds its bound")
    return {
        "schema": "asi.optimizer-geometry.runtime.v1",
        "python_implementation": _runtime_text(
            "Python implementation", platform.python_implementation()
        ),
        "python_version": list(sys.version_info[:3]),
        "byteorder": _runtime_text("byteorder", sys.byteorder),
        "platform": _runtime_text("platform", sys.platform),
        "machine": _runtime_text("machine", platform.machine()),
        "packages": {
            "jax": _runtime_text("JAX version", jax.__version__),
            "jaxlib": _runtime_text(
                "jaxlib version", importlib.metadata.version("jaxlib")
            ),
            "numpy": _runtime_text("NumPy version", np.__version__),
        },
        "default_backend": _runtime_text("JAX backend", jax.default_backend()),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": devices,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_prng_impl": _runtime_text(
            "default PRNG", str(jax.config.jax_default_prng_impl)
        ),
        "jax_default_matmul_precision": _runtime_text(
            "matmul precision", str(jax.config.jax_default_matmul_precision)
        ),
        "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_numpy_dtype_promotion": _runtime_text(
            "dtype promotion", str(jax.config.jax_numpy_dtype_promotion)
        ),
        "jax_numpy_rank_promotion": _runtime_text(
            "rank promotion", str(jax.config.jax_numpy_rank_promotion)
        ),
        "environment": environment,
    }


def _plan_identity(source_sha256: str, runtime_identity: dict[str, object]) -> str:
    payload = {
        "source_sha256": source_sha256,
        "runtime_identity": runtime_identity,
        "protocol": _protocol_payload(),
        "config": dict(FROZEN_GEOMETRY_CONFIG),
        "arm_specs": _ARM_SPECS,
        "control_by_candidate": _CONTROL_BY_CANDIDATE,
    }
    return hashlib.sha256(_canonical_json_bytes_raw(payload)).hexdigest()


def _require_valid(valid: Array, *, name: str) -> None:
    if not bool(valid):
        raise ValueError(f"{name} transaction is invalid")


def _execute_frozen_stream(*, measure_timing: bool) -> dict[str, object]:
    config = FROZEN_GEOMETRY_CONFIG
    stream = _frozen_stream()
    updates = cast(int, config["updates"])
    rows = cast(int, config["rows"])
    columns = cast(int, config["columns"])
    ns_steps = cast(int, config["newton_schulz_steps"])
    dual_steps = cast(int, config["muon_dual_steps"])
    dual_learning_rate = cast(float, config["muon_dual_learning_rate"])
    constraints, basis = _protected_geometry(rows, columns)
    target_updates: list[Array] = []
    for matrix in stream:
        target_update, target_valid = orthogonal_correction_transaction(
            matrix.reshape(-1), basis
        )
        _require_valid(target_valid, name="target projection")
        target_updates.append(target_update.reshape((rows, columns)))
    target = jnp.sum(jnp.stack(target_updates), axis=0)
    if not bool(jnp.all(jnp.isfinite(target))):
        raise ValueError("target accumulation transaction is invalid")
    arm_records: list[dict[str, object]] = []
    for arm, mechanism, mode in _ARM_SPECS:
        started = time.perf_counter_ns() if measure_timing else 0
        state = jnp.zeros((rows, columns), dtype=jnp.float32)
        dual = jnp.zeros((constraints.shape[0],), dtype=jnp.float32)
        processed_updates: list[Array] = []
        for matrix in stream:
            if arm == "muon_ns5_empty_constraints":
                processed, valid = spectral_matrix_sign_transaction(matrix, steps=ns_steps)
            elif arm == "muon_ogd_v2_dual":
                processed, dual, valid = muon_ogd_dual_update_transaction(
                    matrix,
                    constraints,
                    dual,
                    dual_learning_rate=dual_learning_rate,
                    dual_steps=dual_steps,
                    newton_schulz_steps=ns_steps,
                )
            elif arm == "fogo_empty_basis":
                flat_processed, valid = orthogonal_correction_transaction(
                    matrix.reshape(-1), jnp.zeros((0, rows * columns), dtype=matrix.dtype)
                )
                processed = flat_processed.reshape((rows, columns))
            elif arm == "fogo_projection":
                flat_processed, valid = orthogonal_correction_transaction(
                    matrix.reshape(-1), basis
                )
                processed = flat_processed.reshape((rows, columns))
            elif arm == "flad_zero_gradient":
                flat_processed, valid = flad_noise_component_transaction(
                    matrix.reshape(-1), jnp.zeros((rows * columns,), dtype=matrix.dtype)
                )
                processed = flat_processed.reshape((rows, columns))
            else:
                flat_processed, valid = flad_noise_component_transaction(
                    matrix.reshape(-1), basis[0]
                )
                processed = flat_processed.reshape((rows, columns))
            _require_valid(valid, name=f"{arm} update")
            processed_updates.append(processed)
            candidate_state = state + processed
            if not bool(jnp.all(jnp.isfinite(candidate_state))):
                raise ValueError(f"{arm} state transaction is invalid")
            state = candidate_state
        if measure_timing:
            state.block_until_ready()
        elapsed = time.perf_counter_ns() - started if measure_timing else 0
        stacked = jnp.stack(processed_updates)
        final_error = float(jnp.mean(jnp.square(state - target)))
        interference = float(jnp.mean(jnp.abs(stacked[:, 0, 0])))
        mean_update_norm = float(jnp.mean(jnp.linalg.norm(stacked, axis=(1, 2))))
        if not all(math.isfinite(value) for value in (final_error, interference, mean_update_norm)):
            raise ValueError(f"{arm} metric transaction is invalid")
        persistent_bytes = int(state.nbytes)
        if arm == "muon_ogd_v2_dual":
            persistent_bytes += int(dual.nbytes + constraints.nbytes)
        elif mode == "active_basis":
            persistent_bytes += int(basis.nbytes)
        elif mode == "active_gradient":
            persistent_bytes += int(basis[0].nbytes)
        arm_records.append(
            {
                "arm": arm,
                "mechanism": mechanism,
                "mode": mode,
                "metrics": {
                    "final_target_mse": final_error,
                    "mean_protected_interference": interference,
                    "mean_update_frobenius_norm": mean_update_norm,
                },
                "resources": {
                    "persistent_numeric_bytes": persistent_bytes,
                    "observations": updates,
                    "updates": updates,
                    "data_steps": updates,
                    "environment_steps": 0,
                    "model_queries": 0,
                    "timing_ns": elapsed,
                    "timing_qualified": False,
                },
            }
        )
    records_by_arm = {cast(str, record["arm"]): record for record in arm_records}
    comparisons: list[dict[str, object]] = []
    for candidate, control in _CONTROL_BY_CANDIDATE.items():
        candidate_metrics = cast(dict[str, float], records_by_arm[candidate]["metrics"])
        control_metrics = cast(dict[str, float], records_by_arm[control]["metrics"])
        delta = candidate_metrics["final_target_mse"] - control_metrics["final_target_mse"]
        comparisons.append(
            {
                "candidate": candidate,
                "control": control,
                "final_target_mse_delta": delta,
                "outcome": _outcome(delta),
            }
        )
    source_sha256 = _source_sha256()
    runtime_identity = _runtime_identity()
    payload: dict[str, object] = {
        "schema": GEOMETRY_RESULT_SCHEMA,
        "identity": {
            "source_sha256": source_sha256,
            "plan_sha256": _plan_identity(source_sha256, runtime_identity),
            "runtime": runtime_identity,
            "consistency_not_attestation": True,
        },
        "protocol": _protocol_payload(),
        "config": dict(FROZEN_GEOMETRY_CONFIG),
        "stream": {
            "generator": "jax.random.threefry2x32.normal.float32",
            "sha256": _stream_sha256(stream),
            "shape": [updates, rows, columns],
        },
        "policy": {
            "status": "development-only-nonpromoting",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
        },
        "arms": arm_records,
        "comparisons": comparisons,
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_json_bytes_raw(payload)).hexdigest()
    return payload


def run_streaming_matrix_evaluation() -> dict[str, object]:
    """Run and strictly validate the literal frozen development slice."""
    result = _execute_frozen_stream(measure_timing=True)
    validate_streaming_matrix_result(result)
    return result


def _mapping(value: object, *, name: str, maximum_fields: int = 32) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a string-keyed mapping")
    result = cast(dict[object, object], value)
    if len(result) > maximum_fields:
        raise ValueError(f"{name} exceeds the field ceiling")
    if not all(type(key) is str for key in result):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(dict[str, object], result)


def _sequence(value: object, *, name: str, maximum_items: int = 128) -> list[object]:
    if type(value) is not list or len(cast(list[object], value)) > maximum_items:
        raise ValueError(f"{name} must be a list")
    return cast(list[object], value)


def _fields(actual: dict[str, object], expected: tuple[str, ...], *, name: str) -> None:
    if len(actual) != len(expected) or not all(field in actual for field in expected):
        raise ValueError(f"{name} fields do not match the exact schema")


def _same_float(actual: object, expected: object, *, name: str) -> None:
    if type(actual) is not float or type(expected) is not float:
        raise ValueError(f"{name} must be a float")
    if not math.isfinite(actual) or actual != expected:
        raise ValueError(f"{name} does not match the frozen evaluation")


def _exact_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        actual_dict = cast(dict[object, object], actual)
        expected_dict = cast(dict[object, object], expected)
        if (
            len(actual_dict) > 32
            or len(actual_dict) != len(expected_dict)
        ):
            return False
        if not all(type(key) is str for key in actual_dict):
            return False
        if not all(key in actual_dict for key in expected_dict):
            return False
        return all(
            key in actual_dict and _exact_equal(actual_dict[key], value)
            for key, value in expected_dict.items()
        )
    if type(expected) is list:
        actual_list = cast(list[object], actual)
        expected_list = cast(list[object], expected)
        return len(actual_list) <= 128 and len(actual_list) == len(expected_list) and all(
            _exact_equal(left, right)
            for left, right in zip(actual_list, expected_list, strict=True)
        )
    if type(expected) is str:
        actual_text = cast(str, actual)
        try:
            bounded = len(actual_text.encode("utf-8")) <= _MAX_TEXT_BYTES
        except UnicodeEncodeError:
            return False
        return bounded and actual_text == expected
    if type(expected) is int:
        actual_integer = cast(int, actual)
        return -_INT64_MAX <= actual_integer <= _INT64_MAX and actual_integer == expected
    if type(expected) is float:
        actual_float = cast(float, actual)
        return math.isfinite(actual_float) and actual_float == expected
    if type(expected) in (bool, type(None)):
        return actual == expected
    return False


def validate_streaming_matrix_result(result: object) -> None:
    """Fail closed unless ``result`` is an exact current frozen-run record."""
    actual_result = _mapping(result, name="result")
    expected = _execute_frozen_stream(measure_timing=False)
    required_top = (
        "schema",
        "identity",
        "protocol",
        "config",
        "stream",
        "policy",
        "arms",
        "comparisons",
        "result_sha256",
    )
    _fields(actual_result, required_top, name="result")
    if not _exact_equal(actual_result["schema"], GEOMETRY_RESULT_SCHEMA):
        raise ValueError("result fields or schema do not match the frozen protocol")
    for field in ("identity", "protocol", "config", "stream", "policy"):
        actual_mapping = _mapping(actual_result[field], name=field)
        expected_mapping = _mapping(expected[field], name=f"expected.{field}")
        if not _exact_equal(actual_mapping, expected_mapping):
            raise ValueError(f"{field} does not match the frozen protocol")
    arms = _sequence(actual_result["arms"], name="arms")
    expected_arms = _sequence(expected["arms"], name="expected.arms")
    if len(arms) != len(_ARM_SPECS):
        raise ValueError("arms must contain every frozen candidate and control exactly once")
    for index, (raw_arm, raw_expected_arm) in enumerate(zip(arms, expected_arms, strict=True)):
        arm = _mapping(raw_arm, name=f"arms[{index}]")
        expected_arm = _mapping(raw_expected_arm, name=f"expected.arms[{index}]")
        _fields(
            arm,
            ("arm", "mechanism", "mode", "metrics", "resources"),
            name=f"arms[{index}]",
        )
        for field in ("arm", "mechanism", "mode"):
            if not _exact_equal(arm[field], expected_arm[field]):
                raise ValueError(f"arms[{index}].{field} does not match the frozen plan")
        metrics = _mapping(arm["metrics"], name=f"arms[{index}].metrics")
        expected_metrics = _mapping(expected_arm["metrics"], name=f"expected.arms[{index}].metrics")
        _fields(metrics, tuple(expected_metrics), name=f"arms[{index}].metrics")
        for metric, expected_value in expected_metrics.items():
            _same_float(metrics[metric], expected_value, name=f"arms[{index}].metrics.{metric}")
        resources = _mapping(arm["resources"], name=f"arms[{index}].resources")
        expected_resources = _mapping(
            expected_arm["resources"], name=f"expected.arms[{index}].resources"
        )
        _fields(resources, tuple(expected_resources), name=f"arms[{index}].resources")
        for resource, expected_value in expected_resources.items():
            if resource == "timing_ns":
                if (
                    type(resources[resource]) is not int
                    or not 0 <= cast(int, resources[resource]) <= _INT64_MAX
                ):
                    raise ValueError(f"arms[{index}].resources.timing_ns must be non-negative")
            elif resource == "persistent_numeric_bytes" and (
                type(resources[resource]) is not int
                or not 0 <= cast(int, resources[resource]) <= _MAX_RESOURCE_BYTES
            ):
                raise ValueError(
                    f"arms[{index}].resources.persistent_numeric_bytes exceeds its bound"
                )
            elif resource in {
                "observations",
                "updates",
                "data_steps",
                "environment_steps",
                "model_queries",
            } and (
                type(resources[resource]) is not int
                or not 0 <= cast(int, resources[resource]) <= _INT32_MAX
            ):
                raise ValueError(f"arms[{index}].resources.{resource} exceeds its bound")
            elif not _exact_equal(resources[resource], expected_value):
                raise ValueError(f"arms[{index}].resources.{resource} is not canonical")
    comparisons = _sequence(actual_result["comparisons"], name="comparisons")
    expected_comparisons = _sequence(expected["comparisons"], name="expected.comparisons")
    if len(comparisons) != len(expected_comparisons):
        raise ValueError("comparisons must contain every matched A/B pair")
    for index, (raw_comparison, raw_expected_comparison) in enumerate(
        zip(comparisons, expected_comparisons, strict=True)
    ):
        comparison = _mapping(raw_comparison, name=f"comparisons[{index}]")
        expected_comparison = _mapping(
            raw_expected_comparison, name=f"expected.comparisons[{index}]"
        )
        _fields(
            comparison,
            ("candidate", "control", "final_target_mse_delta", "outcome"),
            name=f"comparisons[{index}]",
        )
        for field in ("candidate", "control", "outcome"):
            if not _exact_equal(comparison[field], expected_comparison[field]):
                raise ValueError(f"comparisons[{index}].{field} is not canonical")
        _same_float(
            comparison["final_target_mse_delta"],
            expected_comparison["final_target_mse_delta"],
            name=f"comparisons[{index}].final_target_mse_delta",
        )
    claimed_digest = actual_result["result_sha256"]
    if type(claimed_digest) is not str or len(claimed_digest) != 64:
        raise ValueError("result_sha256 must be lowercase hexadecimal SHA-256")
    unsigned = dict(actual_result)
    del unsigned["result_sha256"]
    actual_digest = hashlib.sha256(_canonical_json_bytes_raw(unsigned)).hexdigest()
    if claimed_digest != actual_digest:
        raise ValueError("result_sha256 does not bind the canonical result")


def canonical_streaming_matrix_result_bytes(result: object) -> bytes:
    """Validate and encode one finite canonical geometry development result."""
    validate_streaming_matrix_result(result)
    return _canonical_json_bytes_raw(result)


def retain_streaming_matrix_result(result: object, *, repository_root: Path) -> Path:
    """Atomically publish a new validated nonpromoting result without replacement."""
    if type(repository_root) is not PosixPath or not repository_root.is_absolute():
        raise ValueError("repository_root must be an exact absolute POSIX Path")
    encoded = canonical_streaming_matrix_result_bytes(result)
    actual = _mapping(result, name="result")
    digest = cast(str, actual["result_sha256"])
    segments = ("outputs", "optimizer_geometry", "development.v1")
    directory = repository_root.joinpath(*segments)
    destination = directory / f"result.{digest}.json"
    temporary_name = f".result.{digest}.tmp"
    destination_name = destination.name
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_descriptor = os.open(repository_root, directory_flags)
    try:
        for segment in segments:
            try:
                os.mkdir(segment, mode=0o755, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(segment, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o444,
            dir_fd=directory_descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(
                temporary_name,
                destination_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        finally:
            os.close(descriptor)
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        read_descriptor = os.open(
            destination_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor
        )
        try:
            with os.fdopen(read_descriptor, "rb", closefd=False) as stream:
                loaded_bytes = stream.read(_MAX_RESULT_BYTES + 1)
        finally:
            os.close(read_descriptor)
        if loaded_bytes != encoded:
            raise RuntimeError("retained geometry result bytes changed during publication")
        loaded = json.loads(loaded_bytes)
        if canonical_streaming_matrix_result_bytes(loaded) != encoded:
            raise RuntimeError("retained geometry result failed strict reload validation")
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination
