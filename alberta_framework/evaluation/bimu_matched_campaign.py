"""Fresh-process execution and validation for the bounded BiMU comparison.

The campaign is permanently nonpromoting and not paper-comparable.  Its
execution gate is deliberately closed in the reviewed source until a separate
authorization change is accepted.  Timing is retained only as telemetry and
never contributes to the paired outcome.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import secrets
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PosixPath
from typing import Any, Final, NoReturn, cast

import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.bimu import (
    _EXAMPLE_ORDER_DOMAIN,
    _INIT_DOMAIN,
    _PRNG_IMPLEMENTATION,
    _dataset_sha256,
    _initialize_state,
    _state_sha256,
    _stream_key,
    build_task_schedule,
    run_bimu_development,
    validate_bimu_result,
)
from alberta_framework.benchmarks.upgd_ipmnist import load_mnist_train
from alberta_framework.evaluation.bimu_matched_nonpromoting import (
    AUTHORIZATION_TRANSITION_APPROVED,
    EXECUTION_AUTHORIZED,
    FROZEN_BIMU_MATCHED_PLAN,
    OUTPUT_NAMESPACE,
    BiMUMatchedDevelopmentPlan,
    _authorization_identity,
    _canonical,
    _dependency_identity,
    _plan_payload,
    _runtime_identity,
    _source_identity,
    frozen_plan_payload,
)

PLAN_DOCUMENT_SCHEMA: Final = "asi.bimu.matched-campaign-plan.v1"
SHARD_SCHEMA: Final = "asi.bimu.matched-campaign-shard.v1"
FAILED_SHARD_SCHEMA: Final = "asi.bimu.matched-campaign-failed-shard.v1"
AGGREGATE_SCHEMA: Final = "asi.bimu.matched-campaign-aggregate.v1"
PROCESS_SCHEMA: Final = "asi.bimu.fresh-process.v1"

MAX_JSON_DEPTH: Final = 64
MAX_JSON_NODES: Final = 50_000
MAX_TEXT_BYTES: Final = 4096
MAX_PLAN_BYTES: Final = 2 * 1024 * 1024
MAX_SHARD_BYTES: Final = 4 * 1024 * 1024
MAX_AGGREGATE_BYTES: Final = 32 * 1024 * 1024
REGISTERED_OUTPUT_ROOT: Final = Path(__file__).resolve().parents[2]

_ARMS: Final = ("memory_off", "bimu")
_MATCHED_COUNTERS: Final = (
    "environment_steps",
    "observations",
    "label_queries",
    "optimizer_updates",
    "optimizer_seen",
    "model_forward_queries",
)
_MATCHED_RESOURCES: Final = (
    "trainable_scalar_count",
    "parameter_numeric_bytes",
    "optimizer_state_numeric_bytes",
    "initial_persistent_numeric_bytes",
    "final_persistent_numeric_bytes",
)
_POLICY: Final = {
    "evidence_class": "bounded_matched_development_comparator",
    "development_only": True,
    "permanently_nonpromoting": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "paper_comparable": False,
    "completed_outcomes_retained": True,
    "ordinary_exception_failure_receipts_enabled": True,
    "post_dispatch_consumed_without_result_tombstone_enabled": True,
    "pre_dispatch_escaped_failure_reservation_released": True,
    "failure_receipt_publication_guaranteed": False,
    "post_dispatch_tombstone_retention_guaranteed": False,
    "process_death_tombstone_retention_guaranteed": False,
    "seed_status": "frozen_exposed_consumed_for_promotion",
}
_TIMING_POLICY: Final = {
    "qualified": False,
    "role": "telemetry_only",
    "used_for_outcome": False,
    "cross_machine_comparison_supported": False,
}


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _require_execution_authorized() -> None:
    if EXECUTION_AUTHORIZED is not True or AUTHORIZATION_TRANSITION_APPROVED is not True:
        raise PermissionError("BiMU campaign execution is not authorized in this source revision")


def _validate_json_tree(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            _fail("JSON input exceeds node ceiling")
        if depth > MAX_JSON_DEPTH:
            _fail("JSON nesting exceeds depth ceiling")
        current_type = type(current)
        if current is None or current_type is bool:
            continue
        if current_type is str:
            if len(cast(str, current).encode("utf-8")) > MAX_TEXT_BYTES:
                _fail("JSON text exceeds byte ceiling")
            continue
        if current_type is int:
            if not -(2**63) <= cast(int, current) <= 2**63 - 1:
                _fail("JSON integer exceeds signed-int64")
            continue
        if current_type is float:
            if not math.isfinite(cast(float, current)):
                _fail("JSON number must be finite")
            continue
        if current_type is list:
            items = cast(list[object], current)
            if len(items) > MAX_JSON_NODES:
                _fail("JSON array exceeds node ceiling")
            pending.extend((item, depth + 1) for item in items)
            continue
        if current_type is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > MAX_JSON_NODES:
                _fail("JSON object exceeds node ceiling")
            for key, item in mapping.items():
                if type(key) is not str or len(key.encode("utf-8")) > MAX_TEXT_BYTES:
                    _fail("JSON object keys must be bounded exact strings")
                pending.append((item, depth + 1))
            continue
        _fail("value is not an exact JSON tree")


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact object")
    resolved = cast(dict[str, object], value)
    if len(resolved) != len(fields) or set(resolved) != fields:
        _fail(f"{label} fields drifted")
    return resolved


def _sha256(value: object) -> str:
    _validate_json_tree(value)
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json_exact_equal(left: object, right: object) -> bool:
    _validate_json_tree(left)
    _validate_json_tree(right)
    return _canonical(left) == _canonical(right)


def digest_without(payload: Mapping[str, object], field: str) -> str:
    """Hash a JSON object after removing its self-digest field."""

    if type(payload) is not dict or type(field) is not str:
        raise TypeError("payload and field must be exact JSON object/string values")
    return _sha256({key: value for key, value in payload.items() if key != field})


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path, *, byte_ceiling: int) -> dict[str, object]:
    """Read one bounded regular JSON file with duplicate-key rejection."""

    if type(path) is not type(Path()) or type(byte_ceiling) is not int or byte_ceiling < 1:
        raise TypeError("path and byte_ceiling must be exact bounded values")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open regular non-symlink JSON input: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail("JSON input must be a regular non-symlink file")
        if before.st_nlink != 1:
            _fail("JSON input must not have a hard-link alias")
        if before.st_size > byte_ceiling:
            _fail("JSON input exceeds byte ceiling")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(byte_ceiling + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) > byte_ceiling:
        _fail("JSON input exceeds byte ceiling")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(raw) != after.st_size:
        _fail("JSON input changed while being read")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_json_object_pairs,
            parse_constant=lambda token: _fail(f"invalid JSON constant: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("JSON input is not one strict document") from exc
    _validate_json_tree(value)
    if type(value) is not dict:
        _fail("JSON document root must be an exact object")
    return cast(dict[str, object], value)


def _current_identity() -> dict[str, object]:
    return {
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "dependencies": _dependency_identity(),
        "authorization": _authorization_identity(),
        "consistency_not_attestation": True,
    }


def build_plan_document() -> dict[str, object]:
    """Build the source-bound literal plan without loading or executing data."""

    payload = frozen_plan_payload()
    document: dict[str, object] = {
        "schema": PLAN_DOCUMENT_SCHEMA,
        "plan": payload,
        "plan_sha256": _sha256(payload),
        "identity": _current_identity(),
        "policy": {
            **_POLICY,
            **_authorization_identity(),
        },
    }
    document["document_sha256"] = _sha256(document)
    validate_plan_document(document)
    return document


def validate_plan_document(value: object) -> dict[str, object]:
    _validate_json_tree(value)
    root = _exact_object(
        value,
        {"schema", "plan", "plan_sha256", "identity", "policy", "document_sha256"},
        "campaign plan",
    )
    expected_plan = frozen_plan_payload()
    if root["schema"] != PLAN_DOCUMENT_SCHEMA or not _json_exact_equal(root["plan"], expected_plan):
        _fail("campaign plan does not match the current literal plan")
    if root["plan_sha256"] != _sha256(expected_plan):
        _fail("campaign plan digest drifted")
    if not _json_exact_equal(root["identity"], _current_identity()):
        _fail("campaign plan identity drifted")
    expected_policy = {**_POLICY, **_authorization_identity()}
    if not _json_exact_equal(root["policy"], expected_policy):
        _fail("campaign plan policy drifted")
    if root["document_sha256"] != digest_without(root, "document_sha256"):
        _fail("campaign plan document digest drifted")
    if len(_canonical(root)) > MAX_PLAN_BYTES:
        _fail("campaign plan exceeds byte ceiling")
    return root


def _validated_arrays(
    train_x: object,
    train_y: object,
    test_x: object,
    test_y: object,
    *,
    plan: BiMUMatchedDevelopmentPlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = plan.candidate_config
    expected = (
        (
            "train_x",
            train_x,
            np.dtype(np.float32),
            (config.train_examples_per_task, config.input_dim),
        ),
        ("train_y", train_y, np.dtype(np.int32), (config.train_examples_per_task,)),
        ("test_x", test_x, np.dtype(np.float32), (config.test_examples_per_task, config.input_dim)),
        ("test_y", test_y, np.dtype(np.int32), (config.test_examples_per_task,)),
    )
    arrays: list[np.ndarray] = []
    byte_count = 0
    for name, value, dtype, shape in expected:
        if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
            _fail(f"{name} does not match the exact campaign shape and dtype")
        byte_count += int(value.nbytes)
        if byte_count > 16 * 1024 * 1024:
            _fail("campaign dataset exceeds its byte ceiling")
        copied = np.array(value, dtype=dtype, order="C", copy=True)
        if dtype == np.dtype(np.float32) and not np.all(np.isfinite(copied)):
            _fail(f"{name} must contain finite values")
        if dtype == np.dtype(np.int32) and (
            np.any(copied < 0) or np.any(copied >= config.n_classes)
        ):
            _fail(f"{name} contains labels outside the campaign class range")
        arrays.append(copied)
    if _dataset_sha256(*arrays) != plan.dataset_sha256:
        _fail("dataset slice does not match the frozen digest")
    return arrays[0], arrays[1], arrays[2], arrays[3]


def load_frozen_bimu_dataset(
    data_home: Path | None = None,
    *,
    plan: BiMUMatchedDevelopmentPlan = FROZEN_BIMU_MATCHED_PLAN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the canonical 60k OpenML materialization and copy its frozen slice."""

    if data_home is not None and type(data_home) is not type(Path()):
        raise TypeError("data_home must be an exact Path or None")
    if type(plan) is not BiMUMatchedDevelopmentPlan:
        raise TypeError("plan must be an exact BiMUMatchedDevelopmentPlan")
    checked = BiMUMatchedDevelopmentPlan(**plan.__dict__)
    full_x, full_y = load_mnist_train(data_home)
    if (
        type(full_x) is not np.ndarray
        or type(full_y) is not np.ndarray
        or full_x.dtype != np.dtype(np.float32)
        or full_y.dtype != np.dtype(np.int32)
        or full_x.shape != (60_000, checked.candidate_config.input_dim)
        or full_y.shape != (60_000,)
    ):
        _fail("canonical OpenML loader did not return the exact 60k materialization")
    count = checked.candidate_config.train_examples_per_task
    test_count = checked.candidate_config.test_examples_per_task
    if count + test_count > 60_000:
        _fail("campaign train/test slices are not disjoint")
    return _validated_arrays(
        np.array(full_x[:count], dtype=np.float32, order="C", copy=True),
        np.array(full_y[:count], dtype=np.int32, order="C", copy=True),
        np.array(full_x[-test_count:], dtype=np.float32, order="C", copy=True),
        np.array(full_y[-test_count:], dtype=np.int32, order="C", copy=True),
        plan=checked,
    )


def _expected_fixed_identities(plan: BiMUMatchedDevelopmentPlan, seed: int) -> tuple[str, str]:
    config = plan.control_config
    root = jr.key(seed, impl=_PRNG_IMPLEMENTATION)
    state = _initialize_state(config, _stream_key(root, _INIT_DOMAIN))
    initial = _state_sha256(state, optimizer_step=0, optimizer_seen=0)
    schedule = hashlib.sha256()
    for task, permutation in enumerate(build_task_schedule(config, seed=seed)):
        order = np.asarray(
            jr.permutation(
                _stream_key(root, _EXAMPLE_ORDER_DOMAIN, task),
                config.train_examples_per_task,
            ),
            dtype=np.int32,
        )
        schedule.update(
            json.dumps(
                {
                    "task": task,
                    "permutation": list(permutation),
                    "example_order": [int(value) for value in order],
                    "query_decisions": [True] * config.train_examples_per_task,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return initial, schedule.hexdigest()


def _process_identity() -> dict[str, object]:
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    stat_path = Path("/proc/self/stat")
    if not boot_path.is_file() or not stat_path.is_file():
        raise RuntimeError("the fresh-process campaign requires Linux /proc identity")
    boot_id = boot_path.read_text(encoding="ascii").strip()
    stat_text = stat_path.read_text(encoding="ascii")
    comm_end = stat_text.rfind(")")
    stat_fields = stat_text[comm_end + 1 :].split() if comm_end >= 0 else []
    if len(stat_fields) < 20 or not stat_fields[19].isdigit():
        raise RuntimeError("cannot resolve Linux process start identity")
    nonce = secrets.token_hex(16)
    process = {
        "schema": PROCESS_SCHEMA,
        "pid": os.getpid(),
        "proc_start_ticks": int(stat_fields[19]),
        "boot_id_sha256": hashlib.sha256(boot_id.encode("ascii")).hexdigest(),
        "invocation_nonce": nonce,
        "fresh_process_required": True,
        "identity_is_not_attestation": True,
    }
    process["execution_instance_id"] = _sha256(process)
    return process


def _arm_config(plan: BiMUMatchedDevelopmentPlan, arm: str) -> Any:
    if type(arm) is not str or arm not in _ARMS:
        _fail("arm is outside the frozen roster")
    return plan.control_config if arm == "memory_off" else plan.candidate_config


def _expected_counters(plan: BiMUMatchedDevelopmentPlan) -> dict[str, int]:
    values = cast(dict[str, object], _plan_payload(plan)["expected_counters_per_arm"])
    return {field: cast(int, values[field]) for field in _MATCHED_COUNTERS}


def _expected_resources(plan: BiMUMatchedDevelopmentPlan) -> dict[str, int]:
    values = cast(dict[str, object], _plan_payload(plan)["expected_resources_per_arm"])
    return {
        field: cast(int, values[field]) for field in (*_MATCHED_RESOURCES, "dataset_numeric_bytes")
    }


def run_bimu_shard(
    arm: str,
    seed: int,
    *,
    data_home: Path | None = None,
    plan_document: object | None = None,
    _loaded_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    _on_first_dispatch: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Execute exactly one authorized arm/seed shard in this process."""

    _require_execution_authorized()
    if type(seed) is not int or seed not in FROZEN_BIMU_MATCHED_PLAN.seeds:
        _fail("seed is outside the frozen roster")
    config = _arm_config(FROZEN_BIMU_MATCHED_PLAN, arm)
    plan_doc = (
        build_plan_document() if plan_document is None else validate_plan_document(plan_document)
    )
    bound_policy = cast(dict[str, object], plan_doc["policy"])
    if (
        bound_policy["execution_authorized"] is not True
        or bound_policy["authorization_transition_approved"] is not True
    ):
        raise PermissionError("the bound campaign plan is not authorized for execution")
    execution_identity = _current_identity()
    if not _json_exact_equal(plan_doc["identity"], execution_identity):
        _fail("campaign identity changed after the plan was validated")
    if _loaded_arrays is None:
        arrays = load_frozen_bimu_dataset(data_home)
    else:
        if type(_loaded_arrays) is not tuple or tuple.__len__(_loaded_arrays) != 4:
            raise TypeError("preloaded campaign arrays must be one exact four-item tuple")
        arrays = _validated_arrays(*_loaded_arrays, plan=FROZEN_BIMU_MATCHED_PLAN)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    if _on_first_dispatch is not None:
        _on_first_dispatch()
    result = run_bimu_development(*arrays, config=config, seed=seed)
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    if _dataset_sha256(*arrays) != FROZEN_BIMU_MATCHED_PLAN.dataset_sha256:
        _fail("campaign dataset changed during shard execution")
    if not _json_exact_equal(_current_identity(), execution_identity):
        _fail("campaign source, runtime, or dependencies changed during shard execution")
    validate_bimu_result(result)
    payload: dict[str, object] = {
        "schema": SHARD_SCHEMA,
        "status": "complete",
        "plan_sha256": plan_doc["plan_sha256"],
        "spec": {"arm": arm, "seed": seed},
        "identity": {
            **execution_identity,
            "dataset_sha256": FROZEN_BIMU_MATCHED_PLAN.dataset_sha256,
            "process": _process_identity(),
        },
        "policy": {**_POLICY, **_authorization_identity()},
        "result": result,
        "resources": {
            **_expected_resources(FROZEN_BIMU_MATCHED_PLAN),
            "numeric_resource_ceiling_bytes": 256 * 1024 * 1024,
            "aggregate_working_set_bytes_claimed": False,
        },
        "timing": {
            "wall_clock_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            **_TIMING_POLICY,
        },
    }
    payload["shard_sha256"] = _sha256(payload)
    validate_bimu_shard(payload)
    return payload


def _finite_nonnegative_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        _fail(f"{label} must be one finite nonnegative float")
    return value


def _validate_process(value: object) -> dict[str, object]:
    process = _exact_object(
        value,
        {
            "schema",
            "pid",
            "proc_start_ticks",
            "boot_id_sha256",
            "invocation_nonce",
            "fresh_process_required",
            "identity_is_not_attestation",
            "execution_instance_id",
        },
        "process identity",
    )
    if process["schema"] != PROCESS_SCHEMA:
        _fail("process identity schema drifted")
    for field in ("pid", "proc_start_ticks"):
        if type(process[field]) is not int or cast(int, process[field]) < 1:
            _fail(f"process identity {field} drifted")
    if not _is_digest(process["boot_id_sha256"]):
        _fail("process boot identity drifted")
    nonce = process["invocation_nonce"]
    if (
        type(nonce) is not str
        or len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        _fail("process invocation nonce drifted")
    if (
        process["fresh_process_required"] is not True
        or process["identity_is_not_attestation"] is not True
    ):
        _fail("process assurance drifted")
    if process["execution_instance_id"] != digest_without(process, "execution_instance_id"):
        _fail("process execution identity digest drifted")
    return process


def validate_bimu_shard(value: object) -> dict[str, object]:
    """Strictly revalidate one complete source-bound shard."""

    _validate_json_tree(value)
    root = _exact_object(
        value,
        {
            "schema",
            "status",
            "plan_sha256",
            "spec",
            "identity",
            "policy",
            "result",
            "resources",
            "timing",
            "shard_sha256",
        },
        "BiMU shard",
    )
    if root["schema"] != SHARD_SCHEMA or root["status"] != "complete":
        _fail("BiMU shard identity or status drifted")
    expected_plan = frozen_plan_payload()
    if root["plan_sha256"] != _sha256(expected_plan):
        _fail("BiMU shard plan digest drifted")
    spec = _exact_object(root["spec"], {"arm", "seed"}, "shard spec")
    arm = cast(str, spec["arm"])
    config = _arm_config(FROZEN_BIMU_MATCHED_PLAN, arm)
    seed = spec["seed"]
    if type(seed) is not int or seed not in FROZEN_BIMU_MATCHED_PLAN.seeds:
        _fail("shard seed is outside the frozen roster")
    identity = _exact_object(
        root["identity"],
        {
            "source_sha256",
            "runtime",
            "dependencies",
            "authorization",
            "consistency_not_attestation",
            "dataset_sha256",
            "process",
        },
        "shard identity",
    )
    expected_identity = {
        **_current_identity(),
        "dataset_sha256": FROZEN_BIMU_MATCHED_PLAN.dataset_sha256,
    }
    if not _json_exact_equal(
        {key: value for key, value in identity.items() if key != "process"},
        expected_identity,
    ):
        _fail("shard execution identity drifted")
    _validate_process(identity["process"])
    if not _json_exact_equal(root["policy"], {**_POLICY, **_authorization_identity()}):
        _fail("shard policy drifted")
    result = root["result"]
    validate_bimu_result(result)
    resolved = cast(dict[str, object], result)
    if resolved["seed"] != seed or not _json_exact_equal(
        resolved["protocol"], config.to_protocol_payload()
    ):
        _fail("shard result does not match its arm/seed spec")
    if resolved["dataset_sha256"] != FROZEN_BIMU_MATCHED_PLAN.dataset_sha256:
        _fail("shard result dataset identity drifted")
    expected_initial, expected_schedule = _expected_fixed_identities(FROZEN_BIMU_MATCHED_PLAN, seed)
    if resolved["initial_state_sha256"] != expected_initial:
        _fail("shard initial-state identity drifted")
    if resolved["schedule_sha256"] != expected_schedule:
        _fail("shard schedule identity drifted")
    counters = cast(dict[str, object], resolved["counters"])
    expected_counters = _expected_counters(FROZEN_BIMU_MATCHED_PLAN)
    if any(counters[field] != expected_counters[field] for field in _MATCHED_COUNTERS):
        _fail("shard counters do not match the literal plan")
    result_resources = cast(dict[str, object], resolved["resources"])
    expected_resources = _expected_resources(FROZEN_BIMU_MATCHED_PLAN)
    if any(result_resources[field] != expected_resources[field] for field in _MATCHED_RESOURCES):
        _fail("shard result resources do not match the literal plan")
    resources = _exact_object(
        root["resources"],
        {
            *_MATCHED_RESOURCES,
            "dataset_numeric_bytes",
            "numeric_resource_ceiling_bytes",
            "aggregate_working_set_bytes_claimed",
        },
        "shard resources",
    )
    if any(resources[field] != expected_resources[field] for field in expected_resources):
        _fail("shard resource binding drifted")
    if (
        resources["numeric_resource_ceiling_bytes"] != 256 * 1024 * 1024
        or resources["aggregate_working_set_bytes_claimed"] is not False
    ):
        _fail("shard resource ceiling policy drifted")
    timing = _exact_object(
        root["timing"],
        {
            "wall_clock_seconds",
            "process_cpu_seconds",
            "qualified",
            "role",
            "used_for_outcome",
            "cross_machine_comparison_supported",
        },
        "shard timing",
    )
    _finite_nonnegative_float(timing["wall_clock_seconds"], "wall timing")
    _finite_nonnegative_float(timing["process_cpu_seconds"], "CPU timing")
    if {key: timing[key] for key in _TIMING_POLICY} != _TIMING_POLICY:
        _fail("shard timing policy drifted")
    if root["shard_sha256"] != digest_without(root, "shard_sha256"):
        _fail("shard digest drifted")
    if len(_canonical(root)) > MAX_SHARD_BYTES:
        _fail("shard exceeds byte ceiling")
    return root


def build_failed_bimu_shard(arm: str, seed: int) -> dict[str, object]:
    """Build one generic receipt for an ordinary failed shard attempt."""

    if type(arm) is not str or arm not in _ARMS:
        _fail("failed shard arm is outside the frozen roster")
    if type(seed) is not int or seed not in FROZEN_BIMU_MATCHED_PLAN.seeds:
        _fail("failed shard seed is outside the frozen roster")
    payload: dict[str, object] = {
        "schema": FAILED_SHARD_SCHEMA,
        "status": "failed",
        "plan_sha256": _sha256(frozen_plan_payload()),
        "spec": {"arm": arm, "seed": seed},
        "identity": {**_current_identity(), "process": _process_identity()},
        "policy": {
            **_POLICY,
            **_authorization_identity(),
            "used_for_outcome": False,
            "retry_authorized": False,
        },
        "failure": {
            "classification": "ordinary_execution_or_validation_failure",
            "exception_type_retained": False,
            "exception_message_retained": False,
            "exception_repr_retained": False,
            "base_exception_retention_guaranteed": False,
        },
    }
    payload["failure_sha256"] = _sha256(payload)
    validate_failed_bimu_shard(payload)
    return payload


def validate_failed_bimu_shard(value: object) -> dict[str, object]:
    """Strictly validate one generic, non-outcome failed-attempt receipt."""

    _validate_json_tree(value)
    root = _exact_object(
        value,
        {
            "schema",
            "status",
            "plan_sha256",
            "spec",
            "identity",
            "policy",
            "failure",
            "failure_sha256",
        },
        "failed BiMU shard",
    )
    if root["schema"] != FAILED_SHARD_SCHEMA or root["status"] != "failed":
        _fail("failed BiMU shard identity or status drifted")
    if root["plan_sha256"] != _sha256(frozen_plan_payload()):
        _fail("failed BiMU shard plan digest drifted")
    spec = _exact_object(root["spec"], {"arm", "seed"}, "failed shard spec")
    if type(spec["arm"]) is not str or spec["arm"] not in _ARMS:
        _fail("failed shard arm is outside the frozen roster")
    if type(spec["seed"]) is not int or spec["seed"] not in FROZEN_BIMU_MATCHED_PLAN.seeds:
        _fail("failed shard seed is outside the frozen roster")
    identity = _exact_object(
        root["identity"],
        {
            "source_sha256",
            "runtime",
            "dependencies",
            "authorization",
            "consistency_not_attestation",
            "process",
        },
        "failed shard identity",
    )
    if not _json_exact_equal(
        {key: item for key, item in identity.items() if key != "process"},
        _current_identity(),
    ):
        _fail("failed shard source or runtime identity drifted")
    _validate_process(identity["process"])
    expected_policy = {
        **_POLICY,
        **_authorization_identity(),
        "used_for_outcome": False,
        "retry_authorized": False,
    }
    if not _json_exact_equal(root["policy"], expected_policy):
        _fail("failed shard policy drifted")
    if not _json_exact_equal(
        root["failure"],
        {
            "classification": "ordinary_execution_or_validation_failure",
            "exception_type_retained": False,
            "exception_message_retained": False,
            "exception_repr_retained": False,
            "base_exception_retention_guaranteed": False,
        },
    ):
        _fail("failed shard disclosure boundary drifted")
    if root["failure_sha256"] != digest_without(root, "failure_sha256"):
        _fail("failed shard digest drifted")
    if len(_canonical(root)) > MAX_SHARD_BYTES:
        _fail("failed shard exceeds byte ceiling")
    return root


def validate_bimu_shard_by_reexecution(
    value: object,
    train_x: object,
    train_y: object,
    test_x: object,
    test_y: object,
) -> dict[str, object]:
    """Reexecute one shard and compare every deterministic result field."""

    _require_execution_authorized()
    replay_identity = _current_identity()
    root = validate_bimu_shard(value)
    arrays = _validated_arrays(
        train_x,
        train_y,
        test_x,
        test_y,
        plan=FROZEN_BIMU_MATCHED_PLAN,
    )
    spec = cast(dict[str, object], root["spec"])
    config = _arm_config(FROZEN_BIMU_MATCHED_PLAN, cast(str, spec["arm"]))
    replay = run_bimu_development(
        *arrays,
        config=config,
        seed=cast(int, spec["seed"]),
    )
    validate_bimu_result(replay)
    reported = cast(dict[str, object], root["result"])
    deterministic_fields = set(replay) - {"timing"}
    if set(reported) - {"timing"} != deterministic_fields or any(
        not _json_exact_equal(reported[field], replay[field]) for field in deterministic_fields
    ):
        _fail("shard result does not match strict seed/arm reexecution")
    if _dataset_sha256(*arrays) != FROZEN_BIMU_MATCHED_PLAN.dataset_sha256:
        _fail("campaign dataset changed during strict shard reexecution")
    if not _json_exact_equal(_current_identity(), replay_identity):
        _fail("campaign source, runtime, or dependencies changed during shard reexecution")
    return root


def _paired_metrics(shards: list[dict[str, object]]) -> dict[str, object]:
    primary: list[float] = []
    secondary: list[float] = []
    for seed in FROZEN_BIMU_MATCHED_PLAN.seeds:
        by_arm = {
            cast(str, cast(dict[str, object], shard["spec"])["arm"]): cast(
                dict[str, object], shard["result"]
            )
            for shard in shards
            if cast(dict[str, object], shard["spec"])["seed"] == seed
        }
        control_metrics = cast(dict[str, object], by_arm["memory_off"]["metrics"])
        candidate_metrics = cast(dict[str, object], by_arm["bimu"]["metrics"])
        primary.append(
            cast(float, candidate_metrics["paper_late_five_test_accuracy"])
            - cast(float, control_metrics["paper_late_five_test_accuracy"])
        )
        secondary.append(
            cast(float, candidate_metrics["asi_whole_stream_online_accuracy"])
            - cast(float, control_metrics["asi_whole_stream_online_accuracy"])
        )
    return {
        "seeds": list(FROZEN_BIMU_MATCHED_PLAN.seeds),
        "primary_metric": "paper_late_five_test_accuracy",
        "primary_deltas": primary,
        "primary_delta_mean": math.fsum(primary) / 3,
        "secondary_metric": "asi_whole_stream_online_accuracy",
        "secondary_deltas": secondary,
        "secondary_delta_mean": math.fsum(secondary) / 3,
    }


def _outcome(paired: dict[str, object]) -> dict[str, object]:
    deltas = cast(list[float], paired["primary_deltas"])
    classification = (
        "supported"
        if all(delta > 0.0 for delta in deltas)
        else "rejected"
        if all(delta <= 0.0 for delta in deltas)
        else "inconclusive"
    )
    rule = cast(dict[str, object], frozen_plan_payload()["paired_outcome_rule"])
    return {
        "classification": classification,
        "rule": rule,
        "scientific_evidence": False,
        "paper_comparable": False,
        "development_selection_only": True,
    }


def _validate_pair_matching(shards: list[dict[str, object]]) -> None:
    for seed in FROZEN_BIMU_MATCHED_PLAN.seeds:
        pair = [shard for shard in shards if cast(dict[str, object], shard["spec"])["seed"] == seed]
        if len(pair) != 2:
            _fail("aggregate roster lacks one complete pair per seed")
        by_arm = {
            cast(str, cast(dict[str, object], shard["spec"])["arm"]): cast(
                dict[str, object], shard["result"]
            )
            for shard in pair
        }
        control = by_arm["memory_off"]
        candidate = by_arm["bimu"]
        for field in ("dataset_sha256", "schedule_sha256", "initial_state_sha256"):
            if control[field] != candidate[field]:
                _fail(f"matched pair {field} drifted")
        control_counters = cast(dict[str, object], control["counters"])
        candidate_counters = cast(dict[str, object], candidate["counters"])
        if any(control_counters[field] != candidate_counters[field] for field in _MATCHED_COUNTERS):
            _fail("matched pair counter drifted")
        control_resources = cast(dict[str, object], control["resources"])
        candidate_resources = cast(dict[str, object], candidate["resources"])
        if any(
            control_resources[field] != candidate_resources[field] for field in _MATCHED_RESOURCES
        ):
            _fail("matched pair resource drifted")


def _aggregate_resources(shards: list[dict[str, object]]) -> dict[str, object]:
    totals = {
        field: sum(
            cast(int, cast(dict[str, object], shard["resources"])[field]) for shard in shards
        )
        for field in (*_MATCHED_RESOURCES, "dataset_numeric_bytes")
    }
    return {
        "totals_across_six_independent_shards": totals,
        "aggregate_working_set_bytes_claimed": False,
    }


def _aggregate_timing(shards: list[dict[str, object]]) -> dict[str, object]:
    return {
        "shard_wall_clock_seconds": [
            cast(dict[str, object], shard["timing"])["wall_clock_seconds"] for shard in shards
        ],
        "shard_process_cpu_seconds": [
            cast(dict[str, object], shard["timing"])["process_cpu_seconds"] for shard in shards
        ],
        **_TIMING_POLICY,
    }


def _summarize_bimu_shards_unchecked(values: object) -> dict[str, object]:
    _validate_json_tree(values)
    if type(values) is not list or len(cast(list[object], values)) != 6:
        _fail("aggregate roster must contain exactly six shards")
    shards = [validate_bimu_shard(item) for item in cast(list[object], values)]
    expected_roster = [(seed, arm) for seed in FROZEN_BIMU_MATCHED_PLAN.seeds for arm in _ARMS]
    observed_roster = [
        (
            cast(dict[str, object], shard["spec"])["seed"],
            cast(dict[str, object], shard["spec"])["arm"],
        )
        for shard in shards
    ]
    if observed_roster != expected_roster:
        _fail("aggregate roster order, membership, or uniqueness drifted")
    execution_ids = [
        cast(
            dict[str, object],
            cast(dict[str, object], shard["identity"])["process"],
        )["execution_instance_id"]
        for shard in shards
    ]
    if len(set(cast(list[str], execution_ids))) != 6:
        _fail("aggregate shards do not have six unique process invocations")
    process_identities = [
        cast(
            dict[str, object],
            cast(dict[str, object], shard["identity"])["process"],
        )
        for shard in shards
    ]
    process_keys = [
        (
            process["boot_id_sha256"],
            process["pid"],
            process["proc_start_ticks"],
        )
        for process in process_identities
    ]
    if len(set(process_keys)) != 6:
        _fail("aggregate requires six independently started fresh processes")
    shared_identity = [
        {
            key: value
            for key, value in cast(dict[str, object], shard["identity"]).items()
            if key != "process"
        }
        for shard in shards
    ]
    if any(not _json_exact_equal(identity, shared_identity[0]) for identity in shared_identity[1:]):
        _fail("aggregate mixes source, runtime, dependency, or dataset identities")
    _validate_pair_matching(shards)
    paired = _paired_metrics(shards)
    aggregate: dict[str, object] = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete",
        "plan": frozen_plan_payload(),
        "plan_sha256": _sha256(frozen_plan_payload()),
        "identity": {
            **shared_identity[0],
            "execution_instance_ids": execution_ids,
            "fresh_process_identity_is_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "shards": shards,
        "paired_metrics": paired,
        "outcome": _outcome(paired),
        "resources": _aggregate_resources(shards),
        "timing": _aggregate_timing(shards),
    }
    aggregate["aggregate_sha256"] = _sha256(aggregate)
    return aggregate


def summarize_bimu_shards(values: object) -> dict[str, object]:
    """Combine the exact six-shard roster and apply the frozen paired rule."""

    aggregate = _summarize_bimu_shards_unchecked(values)
    validate_bimu_aggregate(aggregate)
    return aggregate


def validate_bimu_aggregate(value: object) -> dict[str, object]:
    """Strictly reconstruct a complete aggregate from its retained shards."""

    _validate_json_tree(value)
    root = _exact_object(
        value,
        {
            "schema",
            "status",
            "plan",
            "plan_sha256",
            "identity",
            "policy",
            "shards",
            "paired_metrics",
            "outcome",
            "resources",
            "timing",
            "aggregate_sha256",
        },
        "BiMU aggregate",
    )
    if root["schema"] != AGGREGATE_SCHEMA or root["status"] != "complete":
        _fail("BiMU aggregate identity or status drifted")
    if not _json_exact_equal(root["plan"], frozen_plan_payload()):
        _fail("BiMU aggregate plan drifted")
    shards_value = root["shards"]
    if type(shards_value) is not list:
        _fail("BiMU aggregate shards must be an exact list")
    reconstructed = _summarize_bimu_shards_unchecked(cast(list[object], shards_value))
    for field in (
        "plan_sha256",
        "identity",
        "policy",
        "paired_metrics",
        "outcome",
        "resources",
        "timing",
    ):
        if not _json_exact_equal(root[field], reconstructed[field]):
            _fail(f"BiMU aggregate {field} drifted")
    if root["aggregate_sha256"] != digest_without(root, "aggregate_sha256"):
        _fail("BiMU aggregate digest drifted")
    if len(_canonical(root)) > MAX_AGGREGATE_BYTES:
        _fail("BiMU aggregate exceeds byte ceiling")
    return root


def campaign_path(
    root: Path,
    artifact: str,
    *,
    arm: str | None = None,
    seed: int | None = None,
) -> Path:
    if type(root) is not type(Path()):
        raise TypeError("root must be an exact Path")
    namespace = root / OUTPUT_NAMESPACE
    if artifact == "plan" and arm is None and seed is None:
        return namespace / "plan.json"
    if artifact == "aggregate" and arm is None and seed is None:
        return namespace / "aggregate.json"
    if artifact == "shard" and arm in _ARMS and seed in FROZEN_BIMU_MATCHED_PLAN.seeds:
        return namespace / "shards" / f"seed-{seed}.{arm}.json"
    _fail("artifact does not identify one canonical campaign path")


def _allowed_path(path: Path, root: Path) -> bool:
    candidates = {
        campaign_path(root, "plan"),
        campaign_path(root, "aggregate"),
        *(
            campaign_path(root, "shard", arm=arm, seed=seed)
            for seed in FROZEN_BIMU_MATCHED_PLAN.seeds
            for arm in _ARMS
        ),
    }
    return path in candidates


def _require_registered_root(root: Path) -> None:
    if type(root) is not PosixPath or type(REGISTERED_OUTPUT_ROOT) is not PosixPath:
        raise TypeError("campaign root must be an exact POSIX Path")
    if PosixPath(os.path.abspath(os.fspath(root))) != REGISTERED_OUTPUT_ROOT:
        _fail("campaign output root is not the registered repository root")


def _open_directory_chain(root: Path, segments: Sequence[str], *, create: bool) -> int:
    if type(root) is not PosixPath or not root.is_absolute():
        raise ValueError("directory root must be an exact absolute POSIX Path")
    if (type(segments) is not tuple and type(segments) is not list) or any(
        type(segment) is not str
        or not segment
        or segment in {".", ".."}
        or "/" in segment
        or "\x00" in segment
        for segment in segments
    ):
        raise ValueError("directory segments must be an exact safe sequence")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for segment in segments:
            if create:
                try:
                    os.mkdir(segment, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(segment, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_output_parent(path: Path, *, create: bool) -> tuple[Path, int]:
    """Open one registered parent from the exact pinned repository root."""

    if type(path) is not PosixPath:
        raise TypeError("campaign path must be an exact POSIX Path")
    destination = PosixPath(os.path.abspath(os.fspath(path)))
    candidates = {
        campaign_path(REGISTERED_OUTPUT_ROOT, "plan"),
        campaign_path(REGISTERED_OUTPUT_ROOT, "aggregate"),
        *(
            campaign_path(REGISTERED_OUTPUT_ROOT, "shard", arm=arm, seed=seed)
            for seed in FROZEN_BIMU_MATCHED_PLAN.seeds
            for arm in _ARMS
        ),
    }
    if destination not in candidates:
        _fail("destination is outside the registered campaign namespace")
    segments = tuple(destination.parent.relative_to(REGISTERED_OUTPUT_ROOT).parts)
    return destination, _open_directory_chain(REGISTERED_OUTPUT_ROOT, segments, create=create)


def _link_unnamed_file(file_descriptor: int, parent_descriptor: int, name: str) -> None:
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
    if linkat(file_descriptor, b"", parent_descriptor, os.fsencode(name), 0x1000) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), name)
        raise OSError(error, os.strerror(error), name)


def _require_live_parent(destination: Path, parent_descriptor: int) -> None:
    _, live_descriptor = _open_output_parent(destination, create=False)
    try:
        pinned = os.fstat(parent_descriptor)
        live = os.fstat(live_descriptor)
        if (pinned.st_dev, pinned.st_ino) != (live.st_dev, live.st_ino):
            raise RuntimeError("campaign publication parent changed during publication")
    finally:
        os.close(live_descriptor)


ShardReservation = tuple[Path, int, int, str]


class _CompletedShardAdmissionError(ValueError):
    """A completed payload failed structural or dataset-bound replay admission."""


def _reserve_destination(path: Path, *, root: Path) -> ShardReservation:
    """Reserve one exact artifact before plan/data/replay/execution work."""

    _require_execution_authorized()
    _require_registered_root(root)
    if not _allowed_path(path, root):
        _fail("only one canonical campaign destination may be reserved")
    destination, parent_descriptor = _open_output_parent(path, create=True)
    reservation_name = f".{destination.name}.reservation"
    try:
        try:
            os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to replace existing campaign artifact: {destination}")
        reservation_descriptor = os.open(
            reservation_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
            dir_fd=parent_descriptor,
        )
        marker = b"asi-bimu-matched-campaign-reservation-v1\n"
        written = os.write(reservation_descriptor, marker)
        if written != len(marker):
            raise OSError("campaign reservation write made no progress")
        os.fsync(reservation_descriptor)
        os.fsync(parent_descriptor)
    except BaseException:
        if "reservation_descriptor" in locals():
            _release_shard_reservation(
                (destination, parent_descriptor, reservation_descriptor, reservation_name)
            )
        else:
            os.close(parent_descriptor)
        raise
    return destination, parent_descriptor, reservation_descriptor, reservation_name


def _reserve_shard_destination(path: Path, *, root: Path) -> ShardReservation:
    if path.parent.name != "shards":
        _fail("only one canonical shard destination may be reserved")
    return _reserve_destination(path, root=root)


def _release_shard_reservation(reservation: ShardReservation) -> None:
    destination, parent_descriptor, reservation_descriptor, reservation_name = reservation
    del destination
    try:
        owned = os.fstat(reservation_descriptor)
        try:
            current = os.stat(
                reservation_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                os.unlink(reservation_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
    finally:
        os.close(reservation_descriptor)
        os.close(parent_descriptor)


def _retain_consumed_without_result(reservation: ShardReservation) -> None:
    """Turn the owned visible reservation into a durable no-retry tombstone."""

    destination, parent_descriptor, reservation_descriptor, reservation_name = reservation
    del destination
    marker = b"asi-bimu-matched-campaign-consumed-without-result-v1\n"
    try:
        _require_live_parent(reservation[0], parent_descriptor)
        owned = os.fstat(reservation_descriptor)
        visible = os.stat(
            reservation_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(owned.st_mode)
            or owned.st_nlink != 1
            or (owned.st_dev, owned.st_ino) != (visible.st_dev, visible.st_ino)
        ):
            raise RuntimeError("campaign reservation is not the owned visible inode")
        os.ftruncate(reservation_descriptor, 0)
        os.lseek(reservation_descriptor, 0, os.SEEK_SET)
        written = 0
        while written < len(marker):
            count = os.write(reservation_descriptor, marker[written:])
            if count <= 0:
                raise OSError("campaign tombstone write made no progress")
            written += count
        os.fsync(reservation_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(reservation_descriptor)
        os.close(parent_descriptor)


def publish_json(
    path: Path,
    value: object,
    *,
    root: Path,
    _reservation: ShardReservation | None = None,
    _completed_replay_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
) -> None:
    """Publish one canonical file without replacement and verify its held inode.

    Every final path receives a complete anonymous inode atomically. Shard work
    additionally holds a separate, owned reservation marker until either a
    complete result or a generic failed-attempt receipt is durably published.
    """

    if type(path) is not PosixPath or type(root) is not PosixPath:
        raise TypeError("path and root must be exact POSIX Paths")
    _require_execution_authorized()
    _require_registered_root(root)
    if not _allowed_path(path, root):
        _fail("destination is outside the fixed campaign namespace")
    if path != campaign_path(root, "plan"):
        _require_registered_root(root)

    def validate(candidate: object, *, require_completed_replay: bool) -> None:
        if path == campaign_path(root, "plan"):
            validate_plan_document(candidate)
        elif path == campaign_path(root, "aggregate"):
            validate_bimu_aggregate(candidate)
        else:
            if type(candidate) is dict and candidate.get("schema") == FAILED_SHARD_SCHEMA:
                validate_failed_bimu_shard(candidate)
            elif not require_completed_replay:
                validate_bimu_shard(candidate)
            else:
                try:
                    if (
                        type(_completed_replay_arrays) is not tuple
                        or tuple.__len__(_completed_replay_arrays) != 4
                    ):
                        _fail("completed shard publication requires strict replay arrays")
                    validate_bimu_shard_by_reexecution(candidate, *_completed_replay_arrays)
                except Exception as exc:
                    raise _CompletedShardAdmissionError(
                        "completed shard failed strict dataset-bound admission"
                    ) from exc

    owns_reservation = _reservation is None
    reservation = _reserve_destination(path, root=root) if _reservation is None else _reservation
    destination, parent_descriptor, reservation_descriptor, reservation_name = reservation
    if destination != PosixPath(os.path.abspath(os.fspath(path))):
        _fail("campaign reservation does not match its exact destination")
    _require_live_parent(destination, parent_descriptor)
    reservation_stat = os.fstat(reservation_descriptor)
    visible_reservation = os.stat(
        reservation_name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(reservation_stat.st_mode)
        or reservation_stat.st_nlink != 1
        or (reservation_stat.st_dev, reservation_stat.st_ino)
        != (visible_reservation.st_dev, visible_reservation.st_ino)
    ):
        _fail("campaign reservation is not the owned regular marker")
    file_descriptor: int | None = None
    published_identity: tuple[int, int] | None = None
    try:
        validate(value, require_completed_replay=True)
        raw = _canonical(value) + b"\n"
        shard_path = campaign_path(
            root,
            "shard",
            arm=_ARMS[0],
            seed=FROZEN_BIMU_MATCHED_PLAN.seeds[0],
        )
        _, shard_parent_descriptor = _open_output_parent(shard_path, create=True)
        os.close(shard_parent_descriptor)
        try:
            os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to replace existing campaign artifact: {destination}")
        if not hasattr(os, "O_TMPFILE"):
            raise OSError("immutable publication requires Linux O_TMPFILE support")
        file_descriptor = os.open(
            ".",
            os.O_WRONLY | os.O_CLOEXEC | os.O_TMPFILE,
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(file_descriptor, view[written:])
            if count <= 0:
                raise OSError("campaign publication write made no progress")
            written += count
        os.fsync(file_descriptor)
        os.fchmod(file_descriptor, 0o444)
        source_stat = os.fstat(file_descriptor)
        published_identity = (source_stat.st_dev, source_stat.st_ino)
        try:
            _link_unnamed_file(file_descriptor, parent_descriptor, destination.name)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace existing campaign artifact: {destination}"
            ) from exc
        read_descriptor = os.open(
            destination.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            before = os.fstat(read_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_dev, before.st_ino) != (source_stat.st_dev, source_stat.st_ino)
                or before.st_size != len(raw)
            ):
                raise RuntimeError("published campaign artifact is not the prepared inode")
            loaded = bytearray()
            while len(loaded) <= len(raw):
                chunk = os.read(read_descriptor, min(64 * 1024, len(raw) + 1 - len(loaded)))
                if not chunk:
                    break
                loaded.extend(chunk)
            after = os.fstat(read_descriptor)
        finally:
            os.close(read_descriptor)

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if bytes(loaded) != raw or identity(before) != identity(after):
            raise RuntimeError("published campaign artifact changed during bounded readback")
        try:
            decoded = json.loads(
                loaded,
                object_pairs_hook=_json_object_pairs,
                parse_constant=lambda token: _fail(f"invalid JSON constant: {token}"),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise RuntimeError("published campaign artifact is not strict JSON") from exc
        validate(decoded, require_completed_replay=False)
        os.fsync(parent_descriptor)
        _require_live_parent(destination, parent_descriptor)
    except BaseException:
        if published_identity is not None:
            try:
                visible = os.stat(
                    destination.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (visible.st_dev, visible.st_ino) == published_identity:
                    os.unlink(destination.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if owns_reservation:
            _release_shard_reservation(reservation)


def _load_plan(root: Path) -> dict[str, object]:
    return validate_plan_document(
        load_json_strict(campaign_path(root, "plan"), byte_ceiling=MAX_PLAN_BYTES)
    )


def _validate_namespace(
    root: Path,
    *,
    require_complete_shards: bool,
    require_aggregate: bool,
) -> None:
    namespace = root / OUTPUT_NAMESPACE
    shards_directory = namespace / "shards"
    if not namespace.is_dir() or namespace.is_symlink():
        _fail("campaign namespace must be one real directory")
    expected_top = {"plan.json", "shards"}
    if require_aggregate:
        expected_top.add("aggregate.json")
    observed_top = {entry.name for entry in namespace.iterdir()}
    if observed_top != expected_top:
        _fail("campaign namespace contains missing or unexpected entries")
    if not shards_directory.is_dir() or shards_directory.is_symlink():
        _fail("campaign shard namespace must be one real directory")
    expected_shards = {
        campaign_path(root, "shard", arm=arm, seed=seed).name
        for seed in FROZEN_BIMU_MATCHED_PLAN.seeds
        for arm in _ARMS
    }
    shard_entries = list(shards_directory.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in shard_entries):
        _fail("campaign shard namespace entries must be regular non-symlink files")
    observed_shards = {entry.name for entry in shard_entries}
    if require_complete_shards:
        if observed_shards != expected_shards:
            _fail("campaign shard namespace is not the exact complete roster")
    elif not observed_shards <= expected_shards:
        _fail("campaign shard namespace contains an unexpected entry")


def _load_shards(
    root: Path,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    return [
        (
            validate_bimu_shard(loaded)
            if arrays is None
            else validate_bimu_shard_by_reexecution(loaded, *arrays)
        )
        for seed in FROZEN_BIMU_MATCHED_PLAN.seeds
        for arm in _ARMS
        for loaded in (
            load_json_strict(
                campaign_path(root, "shard", arm=arm, seed=seed),
                byte_ceiling=MAX_SHARD_BYTES,
            ),
        )
    ]


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="permanently nonpromoting bounded BiMU matched campaign"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "summarize", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name in {"summarize", "validate"}:
            command.add_argument("--data-home", type=Path)
    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--root", type=Path, required=True)
    shard.add_argument("--arm", choices=_ARMS, required=True)
    shard.add_argument("--seed", choices=FROZEN_BIMU_MATCHED_PLAN.seeds, required=True, type=int)
    shard.add_argument("--data-home", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = cast(Path, args.root)
    if args.command == "plan":
        _require_execution_authorized()
        document = build_plan_document()
        publish_json(campaign_path(root, "plan"), document, root=root)
        _print_json({"status": "planned", **_authorization_identity()})
        return 0
    if args.command == "run-shard":
        _require_registered_root(root)
        _validate_namespace(
            root,
            require_complete_shards=False,
            require_aggregate=False,
        )
        if EXECUTION_AUTHORIZED is not True or AUTHORIZATION_TRANSITION_APPROVED is not True:
            _print_json(
                {
                    "status": "execution_unauthorized",
                    **_authorization_identity(),
                    "shard_written": False,
                }
            )
            return 2
        destination = campaign_path(root, "shard", arm=args.arm, seed=args.seed)
        reservation = _reserve_shard_destination(destination, root=root)
        reservation_to_cleanup: ShardReservation | None = reservation
        first_dispatch = False
        durable_result_published = False

        def mark_first_dispatch() -> None:
            nonlocal first_dispatch
            first_dispatch = True

        try:
            failed_attempt = False
            try:
                plan = _load_plan(root)
                arrays = load_frozen_bimu_dataset(args.data_home)
                shard = run_bimu_shard(
                    args.arm,
                    args.seed,
                    plan_document=plan,
                    _loaded_arrays=arrays,
                    _on_first_dispatch=mark_first_dispatch,
                )
            except Exception:
                failed_attempt = True
            else:
                try:
                    publish_json(
                        destination,
                        shard,
                        root=root,
                        _reservation=reservation,
                        _completed_replay_arrays=arrays,
                    )
                    durable_result_published = True
                except _CompletedShardAdmissionError:
                    failed_attempt = True
            if failed_attempt:
                failed = build_failed_bimu_shard(args.arm, args.seed)
                publish_json(destination, failed, root=root, _reservation=reservation)
                durable_result_published = True
                _print_json(
                    {
                        "status": "failed",
                        "failure_retained": True,
                        "retry_authorized": False,
                        "shard": str(destination),
                    }
                )
                return 1
        except BaseException:
            if first_dispatch and not durable_result_published:
                reservation_to_cleanup = None
                _retain_consumed_without_result(reservation)
            raise
        finally:
            if reservation_to_cleanup is not None:
                _release_shard_reservation(reservation_to_cleanup)
        _print_json({"status": "complete", "shard": str(destination)})
        return 0
    if args.command == "summarize":
        _require_registered_root(root)
        _validate_namespace(
            root,
            require_complete_shards=True,
            require_aggregate=False,
        )
        destination = campaign_path(root, "aggregate")
        reservation = _reserve_destination(destination, root=root)
        try:
            _load_plan(root)
            arrays = load_frozen_bimu_dataset(args.data_home)
            aggregate = summarize_bimu_shards(_load_shards(root, arrays))
            publish_json(destination, aggregate, root=root, _reservation=reservation)
        finally:
            _release_shard_reservation(reservation)
        _print_json(cast(dict[str, object], aggregate["outcome"]))
        return 0
    if args.command == "validate":
        aggregate_path = campaign_path(root, "aggregate")
        shard_paths = [
            campaign_path(root, "shard", arm=arm, seed=seed)
            for seed in FROZEN_BIMU_MATCHED_PLAN.seeds
            for arm in _ARMS
        ]
        any_shards = any(path.exists() for path in shard_paths)
        if any_shards or aggregate_path.exists():
            _require_registered_root(root)
        _validate_namespace(
            root,
            require_complete_shards=any_shards,
            require_aggregate=aggregate_path.exists(),
        )
        plan = _load_plan(root)
        result: dict[str, object] = {
            "plan_sha256": plan["plan_sha256"],
            **_authorization_identity(),
            "shards": [],
            "aggregate": None,
        }
        replayed_shards: list[dict[str, object]] | None = None
        if any_shards:
            _require_execution_authorized()
            arrays = load_frozen_bimu_dataset(args.data_home)
            replayed_shards = _load_shards(root, arrays)
            result["shards"] = [shard["shard_sha256"] for shard in replayed_shards]
        if aggregate_path.exists():
            aggregate = validate_bimu_aggregate(
                load_json_strict(aggregate_path, byte_ceiling=MAX_AGGREGATE_BYTES)
            )
            if replayed_shards is None or not _json_exact_equal(
                aggregate["shards"], replayed_shards
            ):
                _fail("aggregate shards do not match the dataset-reexecuted shard files")
            result["aggregate"] = aggregate["aggregate_sha256"]
        _print_json(result)
        return 0
    raise AssertionError("argparse returned an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
