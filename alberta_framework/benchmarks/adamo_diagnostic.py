"""Bounded, permanently nonpromoting AdamO diagnostic through the IPMNIST runner.

The adapter reconstructs arXiv:2606.09762v1 because no author-maintained code
was discoverable at qualification time.  It is therefore a comparison target,
not an official-code parity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, mlp_logits
from alberta_framework.core.adamo import ADAMO_PAPER_REVISION, gram_working_bytes

SCHEMA = "asi.adamo_dynamical_isometry_diagnostic.v1"
COMPARISON_ID = "adamo-arxiv-2606.09762v1-ipmnist-adapter-v1"
PAPER_URL = "https://arxiv.org/abs/2606.09762v1"
OFFICIAL_CODE = None
OFFICIAL_CODE_SEARCH_DATE = "2026-08-17"
ARMS = ("adamw_control", "adamo_inert", "adamo_l1e3", "adam_iso_joint_l1e3")
FROZEN_DEVELOPMENT_SEEDS = (15600, 15601, 15602, 15603)
_MAX_DATASET_BYTES = 256 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class DiagnosticProfile:
    name: str
    config: IPMNISTConfig


PROFILES = MappingProxyType({
    "contract-smoke": DiagnosticProfile(
        "contract-smoke",
        IPMNISTConfig(n_tasks=2, task_length=4, input_dim=4, hidden1=3, hidden2=2,
                      n_classes=2),
    ),
    "bounded-development": DiagnosticProfile(
        "bounded-development",
        IPMNISTConfig(n_tasks=8, task_length=64, input_dim=784, hidden1=300, hidden2=150,
                      n_classes=10),
    ),
})


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _tree_sha256(tree: object) -> str:
    digest = hashlib.sha256()
    leaves, structure = jax.tree.flatten(tree)
    digest.update(str(structure).encode("utf-8"))
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _tree_bytes(tree: object) -> int:
    return sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(tree))


def _parameter_count(config: IPMNISTConfig) -> int:
    return (
        config.input_dim * config.hidden1 + config.hidden1
        + config.hidden1 * config.hidden2 + config.hidden2
        + config.hidden2 * config.n_classes + config.n_classes
    )


def _matrix_shapes(config: IPMNISTConfig) -> tuple[tuple[int, int], ...]:
    return (
        (config.input_dim, config.hidden1),
        (config.hidden1, config.hidden2),
        (config.hidden2, config.n_classes),
    )


def _diagnostic(params: Mapping[str, Array], sentinel: Array) -> dict[str, float]:
    materialized = dict(params)
    jacobian = jax.jacrev(lambda x: mlp_logits(materialized, x))(sentinel)
    singular = np.asarray(jnp.linalg.svd(jacobian, compute_uv=False), dtype=np.float64)
    minimum = float(np.min(singular))
    maximum = float(np.max(singular))
    condition_number = (
        1e12 if minimum == 0.0 else min(maximum / minimum, 1e12)
    )
    rms_deviation = float(np.sqrt(np.mean(np.square(singular - 1.0))))
    gram_penalty = 0.0
    for value in materialized.values():
        if value.ndim == 2:
            matrix = np.asarray(value, dtype=np.float64)
            gram = matrix.T @ matrix if matrix.shape[0] >= matrix.shape[1] else matrix @ matrix.T
            gram_penalty += float(np.sum(np.square(gram - np.eye(gram.shape[0]))))
    return {
        "jacobian_min_singular_value": minimum,
        "jacobian_max_singular_value": maximum,
        "jacobian_mean_singular_value": float(np.mean(singular)),
        "jacobian_condition_number_clipped_1e12": condition_number,
        "jacobian_rms_distance_from_one": rms_deviation,
        "weight_gram_penalty": gram_penalty,
    }


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_payload(
    *, result: ScreeningRunResult, diagnostics: list[dict[str, object]],
    persistent_bytes: int, config: IPMNISTConfig,
) -> dict[str, object]:
    observations = config.n_tasks * config.task_length
    parameter_count = _parameter_count(config)
    iso_active = result.config_name in ("adamo_l1e3", "adam_iso_joint_l1e3")
    iso_madds_per_update = sum(
        2 * max(rows, columns) * min(rows, columns) ** 2
        for rows, columns in _matrix_shapes(config)
    ) if iso_active else 0
    logical_compute = (
        4 * observations * parameter_count
        + config.n_tasks * config.n_classes * parameter_count
        + observations * iso_madds_per_update
    )
    return {
        "arm": result.config_name,
        "mechanism": screening_spec(result.config_name).mechanism,
        "hyperparameters": dict(result.hyperparameters),
        "per_task_accuracy": result.per_task_accuracy.tolist(),
        "per_task_loss": result.per_task_loss.tolist(),
        "per_task_plasticity": result.per_task_plasticity.tolist(),
        "post_task_diagnostics": diagnostics,
        "resources": {
            "observations": observations,
            "updates": observations,
            "data_steps": observations,
            "environment_steps": 0,
            "model_queries": 2 * observations + config.n_tasks,
            "jacobian_reverse_rows": config.n_tasks * config.n_classes,
            "parameter_count": parameter_count,
            "persistent_numeric_bytes": persistent_bytes,
            "peak_gram_working_bytes": max(
                gram_working_bytes(shape) for shape in _matrix_shapes(config)
            ),
            "logical_compute_units": logical_compute,
            "logical_compute_definition": (
                "parameter-touch units: 4*observations*parameters + "
                "jacobian_reverse_rows*parameters + active Gram matrix multiply-adds"
            ),
            "timing_seconds": float(result.wall_clock_seconds),
            "timing_is_telemetry_only": True,
        },
    }


def run_adamo_diagnostic(
    inputs: np.ndarray, labels: np.ndarray, *, profile: str, seed: int,
) -> dict[str, object]:
    """Run all four matched arms through the current IPMNIST screening runner."""
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError("profile must name one registered AdamO diagnostic profile")
    resolved_seed = require_jax_seed(seed, name="seed")
    if resolved_seed not in FROZEN_DEVELOPMENT_SEEDS:
        raise ValueError("seed is not in the frozen, consumed development schedule")
    if type(inputs) is not np.ndarray or type(labels) is not np.ndarray:
        raise TypeError("inputs and labels must be exact numpy arrays")
    config = PROFILES[profile].config
    if inputs.dtype != np.float32 or inputs.ndim != 2 or inputs.shape[1] != config.input_dim:
        raise ValueError("inputs must be a finite float32 matrix matching the profile")
    if labels.dtype != np.int32 or labels.shape != (inputs.shape[0],):
        raise ValueError("labels must be an int32 vector matching inputs")
    if inputs.nbytes + labels.nbytes > _MAX_DATASET_BYTES or inputs.shape[0] < config.task_length:
        raise ValueError("dataset is outside the bounded profile contract")
    if not np.isfinite(inputs).all() or np.any(labels < 0) or np.any(labels >= config.n_classes):
        raise ValueError("dataset contains non-finite inputs or out-of-range labels")

    sentinel = jnp.asarray(np.array(inputs[0], copy=True), dtype=jnp.float32)
    arms: list[dict[str, object]] = []
    for arm_name in ARMS:
        diagnostics: list[dict[str, object]] = []
        max_persistent = 0

        def observe(task: int, params: Mapping[str, Array], state: Any) -> None:
            nonlocal max_persistent
            concrete_params = dict(params)
            max_persistent = max(
                max_persistent, _tree_bytes(concrete_params) + _tree_bytes(state)
            )
            diagnostics.append({
                "task_index": task,
                "parameter_sha256": _tree_sha256(concrete_params),
                "learner_state_sha256": _tree_sha256(state),
                **_diagnostic(concrete_params, sentinel),
            })

        result = run_screening_config(
            inputs, labels, screening_spec(arm_name), resolved_seed, config,
            _task_observer=observe,
        )
        arms.append(_arm_payload(
            result=result, diagnostics=diagnostics,
            persistent_bytes=max_persistent, config=config,
        ))

    runner_path = Path(__file__).with_name("ipmnist_screening.py")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "comparison_id": COMPARISON_ID,
        "paper_revision": ADAMO_PAPER_REVISION,
        "paper_url": PAPER_URL,
        "official_code": OFFICIAL_CODE,
        "official_code_search_date": OFFICIAL_CODE_SEARCH_DATE,
        "official_parity_status": "blocked_no_author_maintained_code_located",
        "profile": profile,
        "seed": resolved_seed,
        "frozen_development_seeds": list(FROZEN_DEVELOPMENT_SEEDS),
        "config": asdict(config),
        "dataset": {
            "sha256": _sha256_arrays(inputs, labels),
            "loaded_numeric_bytes": inputs.nbytes + labels.nbytes,
            "rows": inputs.shape[0],
            "materialization": "caller-supplied-float32-inputs-int32-labels-v1",
        },
        "protocol": {
            "arms": list(ARMS),
            "runner": "alberta_framework.benchmarks.ipmnist_screening.run_screening_config",
            "matched_axes": ["dataset", "initialization", "task_permutations", "examples",
                             "seed", "updates", "observations"],
            "learner_boundary_information": [],
            "learner_task_information": ["current_example_label"],
            "diagnostic_information": ["post_task_boundary_index", "fixed_input_row_0"],
            "mechanism_off_reduction": "adamo_inert == adamw_control bit-exact",
            "causal_ablation": "adam_iso_joint_l1e3 mixes task/isometry Adam moments",
        },
        "runtime": {
            "python": platform.python_version(),
            "jax": importlib.metadata.version("jax"),
            "numpy": np.__version__,
            "backend": jax.default_backend(),
        },
        "source": {
            "adapter_sha256": _source_sha256(Path(__file__)),
            "runner_sha256": _source_sha256(runner_path),
        },
        "arms": arms,
        "outcome": "uninterpreted_development_measurement",
        "outcome_retained": True,
        "negative_outcomes_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
    return validate_adamo_diagnostic(payload)


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a built-in integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite built-in number >= {minimum}")
    numeric = cast(int | float, value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{name} must be a finite built-in number >= {minimum}")
    return float(numeric)


def _digest(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_adamo_diagnostic(payload: object) -> dict[str, object]:
    """Fail closed on malformed, drifted, promoting, or unmatched receipts."""
    if type(payload) is not dict:
        raise TypeError("payload must be an exact dict")
    result = cast(dict[str, object], payload)
    required = {
        "schema", "comparison_id", "paper_revision", "paper_url", "official_code",
        "official_code_search_date", "official_parity_status", "profile", "seed",
        "frozen_development_seeds", "config", "dataset", "protocol", "runtime", "source",
        "arms", "outcome", "outcome_retained", "development_only",
        "negative_outcomes_retained", "scientific_promotion_allowed",
    }
    if set(result) != required:
        raise ValueError("payload fields do not match the AdamO v1 schema")
    constants = {
        "schema": SCHEMA, "comparison_id": COMPARISON_ID,
        "paper_revision": ADAMO_PAPER_REVISION, "paper_url": PAPER_URL,
        "official_code": None, "official_code_search_date": OFFICIAL_CODE_SEARCH_DATE,
        "official_parity_status": "blocked_no_author_maintained_code_located",
        "outcome": "uninterpreted_development_measurement", "outcome_retained": True,
        "negative_outcomes_retained": True,
        "development_only": True, "scientific_promotion_allowed": False,
    }
    for key, expected in constants.items():
        if result[key] != expected or type(result[key]) is not type(expected):
            raise ValueError(f"{key} violates the permanent protocol")
    profile = result["profile"]
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError("unknown profile")
    seed = _exact_int(result["seed"], "seed")
    if seed not in FROZEN_DEVELOPMENT_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    if result["frozen_development_seeds"] != list(FROZEN_DEVELOPMENT_SEEDS):
        raise ValueError("frozen seed schedule drift")
    config = PROFILES[profile].config
    if result["config"] != asdict(config):
        raise ValueError("profile configuration drift")
    dataset = result["dataset"]
    if type(dataset) is not dict or set(dataset) != {
        "sha256", "loaded_numeric_bytes", "rows", "materialization"
    }:
        raise ValueError("invalid dataset receipt")
    _digest(dataset["sha256"], "dataset.sha256")
    loaded_bytes = _exact_int(dataset["loaded_numeric_bytes"], "loaded_numeric_bytes", minimum=1)
    rows = _exact_int(dataset["rows"], "rows", minimum=config.task_length)
    if loaded_bytes != rows * (config.input_dim * 4 + 4) or loaded_bytes > _MAX_DATASET_BYTES:
        raise ValueError("dataset byte accounting mismatch")
    if dataset["materialization"] != "caller-supplied-float32-inputs-int32-labels-v1":
        raise ValueError("dataset materialization drift")
    protocol = result["protocol"]
    expected_protocol = {
        "arms": list(ARMS),
        "runner": "alberta_framework.benchmarks.ipmnist_screening.run_screening_config",
        "matched_axes": ["dataset", "initialization", "task_permutations", "examples",
                         "seed", "updates", "observations"],
        "learner_boundary_information": [],
        "learner_task_information": ["current_example_label"],
        "diagnostic_information": ["post_task_boundary_index", "fixed_input_row_0"],
        "mechanism_off_reduction": "adamo_inert == adamw_control bit-exact",
        "causal_ablation": "adam_iso_joint_l1e3 mixes task/isometry Adam moments",
    }
    if type(protocol) is not dict or protocol != expected_protocol:
        raise ValueError("protocol identity or information boundary drift")
    runtime = result["runtime"]
    if type(runtime) is not dict or set(runtime) != {"python", "jax", "numpy", "backend"}:
        raise ValueError("invalid runtime identity")
    expected_runtime = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
    }
    if runtime != expected_runtime:
        raise ValueError("runtime identity does not match the current runtime")
    sources = result["source"]
    if type(sources) is not dict or set(sources) != {"adapter_sha256", "runner_sha256"}:
        raise ValueError("invalid source receipt")
    _digest(sources["adapter_sha256"], "adapter_sha256")
    _digest(sources["runner_sha256"], "runner_sha256")
    if sources["adapter_sha256"] != _source_sha256(Path(__file__)) or sources[
        "runner_sha256"
    ] != _source_sha256(Path(__file__).with_name("ipmnist_screening.py")):
        raise ValueError("source identity does not match the current runner")
    arms = result["arms"]
    if type(arms) is not list or len(arms) != len(ARMS):
        raise ValueError("invalid arm list")
    observations = config.n_tasks * config.task_length
    parameter_count = _parameter_count(config)
    for expected_name, arm in zip(ARMS, arms, strict=True):
        if type(arm) is not dict or set(arm) != {
            "arm", "mechanism", "hyperparameters", "per_task_accuracy", "per_task_loss",
            "per_task_plasticity", "post_task_diagnostics", "resources",
        } or arm.get("arm") != expected_name:
            raise ValueError("arm ordering or identity drift")
        expected_spec = screening_spec(expected_name)
        if arm.get("mechanism") != expected_spec.mechanism or arm.get(
            "hyperparameters"
        ) != expected_spec.hyperparameters:
            raise ValueError("arm mechanism or hyperparameter drift")
        for curve_name in ("per_task_accuracy", "per_task_loss", "per_task_plasticity"):
            curve = arm.get(curve_name)
            if type(curve) is not list or len(curve) != config.n_tasks:
                raise ValueError(f"{expected_name}.{curve_name} has invalid shape")
            for value in curve:
                numeric = _finite(value, f"{expected_name}.{curve_name}")
                if curve_name != "per_task_loss" and numeric > 1.0:
                    raise ValueError(f"{expected_name}.{curve_name} must lie in [0, 1]")
        snapshots = arm.get("post_task_diagnostics")
        if type(snapshots) is not list or len(snapshots) != config.n_tasks:
            raise ValueError("diagnostic task coverage mismatch")
        for index, snapshot in enumerate(snapshots):
            if type(snapshot) is not dict or set(snapshot) != {
                "task_index", "parameter_sha256", "learner_state_sha256",
                "jacobian_min_singular_value", "jacobian_max_singular_value",
                "jacobian_mean_singular_value", "jacobian_condition_number_clipped_1e12",
                "jacobian_rms_distance_from_one", "weight_gram_penalty",
            } or snapshot.get("task_index") != index:
                raise ValueError("diagnostic task index mismatch")
            _digest(snapshot.get("parameter_sha256"), "parameter_sha256")
            _digest(snapshot.get("learner_state_sha256"), "learner_state_sha256")
            for metric in (
                "jacobian_min_singular_value", "jacobian_max_singular_value",
                "jacobian_mean_singular_value", "jacobian_condition_number_clipped_1e12",
                "jacobian_rms_distance_from_one", "weight_gram_penalty",
            ):
                _finite(snapshot.get(metric), metric)
        resources = arm.get("resources")
        if type(resources) is not dict or set(resources) != {
            "observations", "updates", "data_steps", "environment_steps", "model_queries",
            "jacobian_reverse_rows", "parameter_count", "persistent_numeric_bytes",
            "peak_gram_working_bytes", "logical_compute_units", "logical_compute_definition",
            "timing_seconds", "timing_is_telemetry_only",
        }:
            raise ValueError("missing resources")
        exact: dict[str, int] = {
            "observations": observations, "updates": observations,
            "data_steps": observations, "environment_steps": 0,
            "model_queries": 2 * observations + config.n_tasks,
            "jacobian_reverse_rows": config.n_tasks * config.n_classes,
            "parameter_count": parameter_count,
            "persistent_numeric_bytes": 12 * parameter_count + 120,
            "peak_gram_working_bytes": max(
                gram_working_bytes(shape) for shape in _matrix_shapes(config)
            ),
        }
        iso_active = expected_name in ("adamo_l1e3", "adam_iso_joint_l1e3")
        iso_madds = sum(
            2 * max(rows_count, columns_count) * min(rows_count, columns_count) ** 2
            for rows_count, columns_count in _matrix_shapes(config)
        ) if iso_active else 0
        exact["logical_compute_units"] = (
            4 * observations * parameter_count
            + config.n_tasks * config.n_classes * parameter_count
            + observations * iso_madds
        )
        for key, expected_count in exact.items():
            if resources.get(key) != expected_count or type(resources.get(key)) is not int:
                raise ValueError(f"{expected_name}.{key} accounting mismatch")
        if resources.get("logical_compute_definition") != (
            "parameter-touch units: 4*observations*parameters + "
            "jacobian_reverse_rows*parameters + active Gram matrix multiply-adds"
        ):
            raise ValueError("logical compute definition drift")
        _finite(resources.get("timing_seconds"), "timing_seconds")
        if resources.get("timing_is_telemetry_only") is not True:
            raise ValueError("timing may only be telemetry")
    control = cast(dict[str, object], arms[0])
    inert = cast(dict[str, object], arms[1])
    for key in ("per_task_accuracy", "per_task_loss", "per_task_plasticity",
                "post_task_diagnostics"):
        if inert[key] != control[key]:
            raise ValueError("AdamO inert arm does not reduce bit-exactly to AdamW")
    return result


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_DATASET_BYTES:
        raise ValueError("dataset must be a bounded, non-symlink NPZ file")
    try:
        with zipfile.ZipFile(path) as zip_archive:
            members = zip_archive.infolist()
            if (
                len(members) != 2
                or {member.filename for member in members} != {"inputs.npy", "labels.npy"}
                or sum(member.file_size for member in members) > _MAX_DATASET_BYTES
                or any(member.is_dir() for member in members)
            ):
                raise ValueError("dataset NPZ members exceed the bounded exact contract")
    except zipfile.BadZipFile as exc:
        raise ValueError("dataset must be a valid NPZ archive") from exc
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"inputs", "labels"}:
            raise ValueError("dataset NPZ must contain exactly inputs and labels")
        return np.array(archive["inputs"], copy=True), np.array(archive["labels"], copy=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="contract-smoke")
    parser.add_argument("--seed", type=int, default=FROZEN_DEVELOPMENT_SEEDS[0])
    args = parser.parse_args(argv)
    if args.catalog:
        if args.dataset is not None:
            parser.error("--catalog and --dataset are mutually exclusive")
        print(json.dumps({
            "schema": SCHEMA, "paper_revision": ADAMO_PAPER_REVISION,
            "official_code": None, "profiles": {name: asdict(value.config)
                                                   for name, value in PROFILES.items()},
            "arms": list(ARMS), "frozen_development_seeds": list(FROZEN_DEVELOPMENT_SEEDS),
            "development_only": True, "negative_outcomes_retained": True,
            "scientific_promotion_allowed": False,
        }, sort_keys=True))
        return 0
    if args.dataset is None:
        parser.error("--dataset is required unless --catalog is used")
    inputs, labels = _load_dataset(args.dataset)
    print(json.dumps(run_adamo_diagnostic(inputs, labels, profile=args.profile, seed=args.seed),
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
