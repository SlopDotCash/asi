"""Prospective, hard-disabled matched NaP campaign for issue #1564."""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.nap_ipmnist import (
    ARM_IDS,
    CAMPAIGN_RESERVED_SEEDS,
    PAPER_IDENTITY,
    PAPER_REVISION,
    NaPCatalogEntry,
    _run_comparator_for_seeds,
    _schedule_sha,
)
from alberta_framework.benchmarks.nap_ipmnist import SCHEMA as RESULT_SCHEMA
from alberta_framework.benchmarks.nap_ipmnist import (
    _runtime_identity as _comparator_runtime_identity,
)
from alberta_framework.benchmarks.plasticity_diagnostics import (
    INPUT_DIM,
    N_CLASSES,
    PROFILES,
    _arrays,
    _dataset_sha,
    _init_state,
    _schedule,
    _state_sha256,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    default_openml_data_home,
    load_mnist_train,
)

SCHEMA: Final = "asi.nap.matched-development.v1"
TEST_ONLY_SEEDS: Final = (15_640, 15_641, 15_642, 15_643)
PROFILE_ID: Final = "bounded-development"
DATASET_SHA256: Final = "234322a369029211eb4555087fc5448c972215e4a50dc4e4d8a21b5a3f8d4d9a"
_REVIEWED_EXECUTION_TRANSITION: Final = False
_EXECUTION_AUTHORIZED: Final = False
_ROOT: Final = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Final = _ROOT / "outputs/nap_matched/v1/report.json"
_MAX_NODES: Final = 200_000
_MAX_DEPTH: Final = 20
_MAX_STRING_BYTES: Final = 16_384
_MAX_REPORT_BYTES: Final = 64 * 1024 * 1024
_MAX_DATASET_BYTES: Final = 256 * 1024 * 1024


def _source_identity() -> dict[str, str]:
    paths = (
        "alberta_framework/benchmarks/nap_ipmnist.py",
        "alberta_framework/benchmarks/plasticity_comparators.py",
        "alberta_framework/benchmarks/plasticity_diagnostics.py",
        "alberta_framework/evaluation/nap_matched_campaign.py",
        "pyproject.toml",
        "uv.lock",
    )
    return {path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest() for path in paths}


def _runtime_identity() -> dict[str, object]:
    devices = tuple(jax.devices())
    if not 1 <= len(devices) <= 64:
        raise RuntimeError("NaP runtime device inventory is out of bounds")
    if (
        str(jax.config.jax_default_prng_impl) != "threefry2x32"
        or int(jax.config.jax_random_seed_offset) != 0
    ):
        raise RuntimeError("NaP campaign requires explicit Threefry runtime configuration")
    return {
        "python": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "platform": [platform.system(), platform.release(), platform.machine()],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "scikit-learn")
        },
        "jax": {
            "backend": jax.default_backend(),
            "prng": str(jax.config.jax_default_prng_impl),
            "seed_offset": int(jax.config.jax_random_seed_offset),
            "x64": bool(jax.config.jax_enable_x64),
            "devices": [
                [int(device.id), str(device.platform), str(device.device_kind)]
                for device in devices
            ],
        },
    }


def frozen_plan() -> dict[str, object]:
    profile = PROFILES[PROFILE_ID]
    steps = profile.n_tasks * profile.examples_per_task
    dispatches = 2 * len(CAMPAIGN_RESERVED_SEEDS)
    state_bytes = sum(
        np.asarray(leaf).nbytes
        for leaf in jax.tree.leaves(
            _init_state(jr.key(0, impl="threefry2x32"), profile.hidden_width)
        )
    )
    return {
        "schema": "asi.nap.matched-plan.v1",
        "issue": "SlopDotCash/asi#1564",
        "paper_revision": PAPER_REVISION,
        "paper_identity": PAPER_IDENTITY,
        "official_method_source": "not-disclosed-or-located",
        "secondary_source": {
            "repository": "https://github.com/RLE-Foundation/Plasticine.git",
            "commit": "aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8",
            "official": False,
        },
        "protocol_differences": [
            "cumulative input-permuted MNIST rather than random-label CIFAR or ALE",
            "bounded MLP rather than paper CNN/VGG or Rainbow",
            "per-example SGD and fixed unlearned normalization",
            "biases retained and final output layer not projected",
            "paper scale-offset and learning-rate schedules are absent",
        ],
        "profile_id": PROFILE_ID,
        "profile": dataclasses.asdict(profile),
        "seeds": list(CAMPAIGN_RESERVED_SEEDS),
        "test_only_seeds": list(TEST_ONLY_SEEDS),
        "arms": list(ARM_IDS),
        "dataset": {
            "provider": "openml",
            "name": "mnist_784",
            "version": 1,
            "rows": 60_000,
            "materialization": "float32-zero-one-int32-labels",
            "sha256": DATASET_SHA256,
        },
        "matched_axes": [
            "seed", "dataset", "schedule", "initialization", "observations",
            "updates", "allowed_boundary_information", "allowed_task_information",
        ],
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "primary_paired_question": {
            "candidate": "nap",
            "control": "nap_mechanism_off",
            "metric": "mean_task_accuracy",
            "decision_rule": "advance iff mean_delta > 0 and at least 4 of 5 deltas > 0",
            "other_arms": "descriptive_only",
        },
        "resources": {
            "comparator_dispatches": dispatches,
            "arm_dispatches": dispatches * len(ARM_IDS),
            "observations": dispatches * len(ARM_IDS) * steps,
            "data_steps": dispatches * len(ARM_IDS) * steps,
            "parameter_updates": dispatches * len(ARM_IDS) * steps,
            "model_queries": dispatches * len(ARM_IDS) * steps * 3,
            "environment_steps": 0,
            "per_arm_state_persistent_bytes": state_bytes,
            "peak_comparator_persistent_bytes": len(ARM_IDS) * state_bytes + 16,
            "dataset_bytes_limit": _MAX_DATASET_BYTES,
            "timing_is_telemetry_only": True,
        },
        "identity_policy": "exact source/dependency/runtime/JAX/device/dataset/schedule",
        "execution": {"reviewed_transition": False, "authorized": False},
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_results_retained": True,
            "completed_result_exists": False,
        },
        "output_path": "outputs/nap_matched/v1/report.json",
    }


def _bounded(value: object) -> object:
    nodes = 0
    seen: set[int] = set()

    def visit(item: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("NaP campaign JSON exceeds its structure bound")
        actual = type(item)
        if actual is type(None) or actual is bool:
            return item
        if actual is int:
            if not -(2**63) <= cast(int, item) <= 2**63 - 1:
                raise ValueError("NaP campaign JSON integer is out of bounds")
            return item
        if actual is float:
            if not math.isfinite(cast(float, item)):
                raise ValueError("NaP campaign JSON contains a non-finite float")
            return item
        if actual is str:
            if len(cast(str, item).encode("utf-8")) > _MAX_STRING_BYTES:
                raise ValueError("NaP campaign JSON contains oversized text")
            return item
        if actual is not dict and actual is not list:
            raise ValueError("NaP campaign records require exact JSON values")
        identity = id(item)
        if identity in seen:
            raise ValueError("NaP campaign JSON contains aliased containers")
        seen.add(identity)
        if actual is list:
            return [visit(child, depth + 1) for child in cast(list[object], item)]
        result: dict[str, object] = {}
        for key, child in cast(dict[object, object], item).items():
            if type(key) is not str:
                raise ValueError("NaP campaign JSON keys must be exact strings")
            result[key] = visit(child, depth + 1)
        return result

    return visit(value, 0)


def _canonical(value: object) -> bytes:
    encoded = json.dumps(
        _bounded(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("NaP campaign report exceeds its byte bound")
    return encoded


def _sign(report: dict[str, object]) -> None:
    report.pop("sha256", None)
    report["sha256"] = hashlib.sha256(_canonical(report)).hexdigest()


def _load_canonical_dataset(data_home: Path) -> tuple[np.ndarray, np.ndarray]:
    images, labels = load_mnist_train(data_home)
    images = np.ascontiguousarray((images + np.float32(1.0)) / np.float32(2.0))
    images, labels = _arrays(images, labels)
    if images.nbytes + labels.nbytes > _MAX_DATASET_BYTES:
        raise ValueError("NaP canonical dataset exceeds its byte ceiling")
    if _dataset_sha(images, labels) != DATASET_SHA256:
        raise ValueError("NaP canonical OpenML dataset identity drifted")
    return images, labels


def _result_payload(
    images: np.ndarray, labels: np.ndarray, seed: int, profile_id: str
) -> dict[str, object]:
    result = _run_comparator_for_seeds(
        images,
        labels,
        seed=seed,
        profile_id=profile_id,
        allowed_seeds=(
            CAMPAIGN_RESERVED_SEEDS if seed in CAMPAIGN_RESERVED_SEEDS else TEST_ONLY_SEEDS
        ),
    )
    return cast(
        dict[str, object],
        json.loads(json.dumps(dataclasses.asdict(result), allow_nan=False)),
    )


def _mean_accuracy(result: dict[str, object], arm_id: str) -> float:
    arms = cast(list[dict[str, object]], result["arms"])
    arm = next(item for item in arms if item["arm_id"] == arm_id)
    values = cast(list[float], arm["task_accuracy"])
    return math.fsum(values) / len(values)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [
        {
            "seed": cast(int, row["seed"]),
            "nap_minus_off": _mean_accuracy(cast(dict[str, object], row["result"]), "nap")
            - _mean_accuracy(cast(dict[str, object], row["result"]), "nap_mechanism_off"),
        }
        for row in rows
    ]
    values = [item["nap_minus_off"] for item in deltas]
    mean = math.fsum(values) / len(values)
    positives = sum(value > 0.0 for value in values)
    outcome = (
        "advance_for_nonpromoting_followup"
        if mean > 0.0 and positives >= 4
        else "do_not_advance"
    )
    return {
        "paired_deltas": deltas,
        "mean_delta": mean,
        "positive_seed_count": positives,
        "outcome": outcome,
    }


def _run(
    images: object,
    labels: object,
    *,
    seeds: tuple[int, ...],
    profile_id: str,
    on_dispatch: object = None,
) -> dict[str, object]:
    checked_images, checked_labels = _arrays(images, labels)
    if seeds not in (CAMPAIGN_RESERVED_SEEDS, TEST_ONLY_SEEDS):
        raise ValueError("NaP campaign seed roster drifted")
    if profile_id not in PROFILES:
        raise ValueError("NaP campaign profile drifted")
    source = _source_identity()
    runtime = _runtime_identity()
    dataset = _dataset_sha(checked_images, checked_labels)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        if not rows and on_dispatch is not None:
            if not callable(on_dispatch):
                raise TypeError("NaP dispatch callback must be callable")
            on_dispatch()
        root = jr.key(seed, impl="threefry2x32")
        _next_key, init_key = jr.split(root)
        rows.append(
            {
                "seed": seed,
                "initial_state_sha256": _state_sha256(
                    _init_state(init_key, PROFILES[profile_id].hidden_width)
                ),
                "result": _result_payload(
                    checked_images, checked_labels, seed, profile_id
                ),
            }
        )
    if source != _source_identity() or runtime != _runtime_identity():
        raise RuntimeError("NaP campaign identity changed during execution")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "plan": frozen_plan(),
        "identity": {"source": source, "runtime": runtime, "dataset_sha256": dataset},
        "profile_id": profile_id,
        "seeds": list(seeds),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_results_retained": True,
        },
    }
    _sign(report)
    validate_report(
        report, checked_images, checked_labels, seeds=seeds, profile_id=profile_id,
        reexecute=False
    )
    return report


def _run_for_test(images: object, labels: object) -> dict[str, object]:
    return _run(images, labels, seeds=TEST_ONLY_SEEDS, profile_id="contract-smoke")


def _validate_result_payload(value: object, seed: int, profile_id: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("NaP campaign result must be an exact object")
    result = cast(dict[str, object], value)
    expected_keys = {
        "schema", "catalog", "profile_id", "profile", "seed", "dataset_sha256",
        "schedule_sha256", "source_sha256", "dependency_source_sha256",
        "nap_project_dependency_source_sha256", "runtime_identity", "task_protocol",
        "labels_permuted", "task_boundaries_visible_to_learner", "task_ids_visible_to_learner",
        "observations_matched_before_causal_divergence", "arms",
        "negative_results_must_be_retained", "development_only",
        "scientific_promotion_allowed", "paper_parity_claimed",
    }
    if set(result) != expected_keys or result["schema"] != RESULT_SCHEMA:
        raise ValueError("NaP campaign result schema drifted")
    if result["seed"] != seed or result["profile_id"] != profile_id:
        raise ValueError("NaP campaign result roster drifted")
    profile = PROFILES[profile_id]
    expected_static = {
        "catalog": json.loads(json.dumps(dataclasses.asdict(NaPCatalogEntry()))),
        "profile": dataclasses.asdict(profile),
        "source_sha256": hashlib.sha256(
            (_ROOT / "alberta_framework/benchmarks/nap_ipmnist.py").read_bytes()
        ).hexdigest(),
        "dependency_source_sha256": hashlib.sha256(
            (_ROOT / "alberta_framework/benchmarks/plasticity_diagnostics.py").read_bytes()
        ).hexdigest(),
        "nap_project_dependency_source_sha256": hashlib.sha256(
            (_ROOT / "alberta_framework/benchmarks/plasticity_comparators.py").read_bytes()
        ).hexdigest(),
        "runtime_identity": list(_comparator_runtime_identity()),
        "task_protocol": "cumulative-input-permuted-mnist",
        "labels_permuted": False,
        "task_boundaries_visible_to_learner": False,
        "task_ids_visible_to_learner": False,
        "observations_matched_before_causal_divergence": True,
    }
    if any(result[name] != expected for name, expected in expected_static.items()):
        raise ValueError("NaP campaign result source/runtime/protocol identity drifted")
    if (
        result["development_only"] is not True
        or result["scientific_promotion_allowed"] is not False
        or result["negative_results_must_be_retained"] is not True
        or result["paper_parity_claimed"] is not False
    ):
        raise ValueError("NaP campaign result policy drifted")
    arms = result["arms"]
    if type(arms) is not list or len(arms) != len(ARM_IDS):
        raise ValueError("NaP campaign arm roster drifted")
    steps = profile.n_tasks * profile.examples_per_task
    forward_macs = (
        INPUT_DIM * profile.hidden_width
        + profile.hidden_width * profile.hidden_width
        + profile.hidden_width * N_CLASSES
    )
    state_bytes = sum(
        np.asarray(leaf).nbytes
        for leaf in jax.tree.leaves(
            _init_state(jr.key(0, impl="threefry2x32"), profile.hidden_width)
        )
    )
    for index, raw_arm in enumerate(cast(list[object], arms)):
        if type(raw_arm) is not dict:
            raise ValueError("NaP campaign arm must be an exact object")
        arm = cast(dict[str, object], raw_arm)
        arm_fields = {
            "arm_id", "normalization_enabled", "projection_enabled", "task_accuracy",
            "task_loss", "dead_unit_fraction", "effective_rank", "initial_hidden_norms",
            "final_hidden_norms", "final_state_sha256", "receipt",
        }
        if set(arm) != arm_fields:
            raise ValueError("NaP campaign arm fields drifted")
        if arm.get("arm_id") != ARM_IDS[index]:
            raise ValueError("NaP campaign arm identity drifted")
        expected_flags = {
            "sgd_current_control": (False, False),
            "nap_mechanism_off": (False, False),
            "normalization_only": (True, False),
            "projection_only": (False, True),
            "nap": (True, True),
        }[ARM_IDS[index]]
        if (arm.get("normalization_enabled"), arm.get("projection_enabled")) != expected_flags:
            raise ValueError("NaP campaign arm mechanism flags drifted")
        for metric in ("task_accuracy", "task_loss", "dead_unit_fraction", "effective_rank"):
            curve = arm.get(metric)
            if type(curve) is not list or len(cast(list[object], curve)) != profile.n_tasks:
                raise ValueError("NaP campaign metric shape drifted")
            for item in cast(list[object], curve):
                if type(item) is not float or not math.isfinite(item):
                    raise ValueError("NaP campaign metric domain drifted")
                if metric in {"task_accuracy", "dead_unit_fraction"} and not (
                    0.0 <= item <= 1.0
                ):
                    raise ValueError("NaP campaign probability metric drifted")
                if metric in {"task_loss", "effective_rank"} and item < 0.0:
                    raise ValueError("NaP campaign nonnegative metric drifted")
        for name in ("initial_hidden_norms", "final_hidden_norms"):
            norms = arm.get(name)
            if (
                type(norms) is not list
                or len(cast(list[object], norms)) != 2
                or any(
                    type(item) is not float
                    or not math.isfinite(item)
                    or item <= 0.0
                    for item in cast(list[object], norms)
                )
            ):
                raise ValueError("NaP campaign hidden-norm identity drifted")
        receipt = arm.get("receipt")
        if type(receipt) is not dict:
            raise ValueError("NaP campaign receipt drifted")
        receipt_map = cast(dict[str, object], receipt)
        receipt_fields = {
            "data_steps", "observations", "data_bytes_read", "training_model_queries",
            "diagnostic_model_queries", "model_queries", "parameter_updates",
            "normalization_queries", "normalization_elements", "projection_events",
            "projected_tensor_queries", "projected_elements", "logical_forward_macs",
            "logical_gradient_macs", "logical_auxiliary_scalar_ops",
            "state_persistent_bytes", "projection_target_persistent_bytes", "elapsed_ns",
            "timing_telemetry_only",
        }
        if set(receipt_map) != receipt_fields:
            raise ValueError("NaP campaign receipt fields drifted")
        normalize = ARM_IDS[index] in {"normalization_only", "nap"}
        project = ARM_IDS[index] in {"projection_only", "nap"}
        normalization_queries = 6 * steps if normalize else 0
        normalization_elements = normalization_queries * profile.hidden_width
        projection_events = steps if project else 0
        projected_elements = projection_events * (
            INPUT_DIM * profile.hidden_width + profile.hidden_width * profile.hidden_width
        )
        exact_receipt = {
            "data_steps": steps,
            "observations": steps,
            "data_bytes_read": steps * (INPUT_DIM * 4 + 4),
            "parameter_updates": steps,
            "training_model_queries": 2 * steps,
            "diagnostic_model_queries": steps,
            "model_queries": 3 * steps,
            "normalization_queries": normalization_queries,
            "normalization_elements": normalization_elements,
            "projection_events": projection_events,
            "projected_tensor_queries": 2 * projection_events,
            "projected_elements": projected_elements,
            "logical_forward_macs": 3 * steps * forward_macs,
            "logical_gradient_macs": 2 * steps * forward_macs,
            "logical_auxiliary_scalar_ops": (
                normalization_elements * 5 + projected_elements * 3
            ),
            "state_persistent_bytes": state_bytes,
            "projection_target_persistent_bytes": 8 if project else 0,
            "timing_telemetry_only": True,
        }
        for name, expected in exact_receipt.items():
            if (
                type(receipt_map.get(name)) is not type(expected)
                or receipt_map.get(name) != expected
            ):
                raise ValueError("NaP campaign resource receipt drifted")
        elapsed = receipt_map.get("elapsed_ns")
        if type(elapsed) is not int or elapsed < 0:
            raise ValueError("NaP campaign timing receipt drifted")
        digest = arm["final_state_sha256"]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("NaP campaign final-state identity drifted")
    current, mechanism_off, *_rest = cast(list[dict[str, object]], arms)
    for name in (
        "task_accuracy", "task_loss", "dead_unit_fraction", "effective_rank",
        "initial_hidden_norms", "final_hidden_norms", "final_state_sha256",
    ):
        if current[name] != mechanism_off[name]:
            raise ValueError("NaP campaign mechanism-off parity drifted")
    current_receipt = dict(cast(dict[str, object], current["receipt"]))
    off_receipt = dict(cast(dict[str, object], mechanism_off["receipt"]))
    current_receipt["elapsed_ns"] = 0
    off_receipt["elapsed_ns"] = 0
    if current_receipt != off_receipt:
        raise ValueError("NaP campaign mechanism-off resource parity drifted")
    return result


def _without_timing(value: dict[str, object]) -> dict[str, object]:
    result = cast(dict[str, object], _bounded(value))
    for arm in cast(list[dict[str, object]], result["arms"]):
        cast(dict[str, object], arm["receipt"])["elapsed_ns"] = 0
    return result


def validate_report(
    value: object,
    images: object,
    labels: object,
    *,
    seeds: tuple[int, ...],
    profile_id: str,
    reexecute: bool,
) -> None:
    if seeds == CAMPAIGN_RESERVED_SEEDS and reexecute and (
        _REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True
    ):
        raise PermissionError("NaP campaign execution is not authorized")
    checked_images, checked_labels = _arrays(images, labels)
    report = cast(dict[str, object], _bounded(value))
    report_fields = {
        "schema", "plan", "identity", "profile_id", "seeds", "rows", "aggregate",
        "policy", "sha256",
    }
    if set(report) != report_fields:
        raise ValueError("NaP campaign report fields drifted")
    if report["schema"] != SCHEMA or report["plan"] != frozen_plan():
        raise ValueError("NaP campaign plan drifted")
    if report["profile_id"] != profile_id or report["seeds"] != list(seeds):
        raise ValueError("NaP campaign roster drifted")
    identity = {
        "source": _source_identity(),
        "runtime": _runtime_identity(),
        "dataset_sha256": _dataset_sha(checked_images, checked_labels),
    }
    if report["identity"] != identity:
        raise ValueError("NaP campaign identity drifted")
    rows = report["rows"]
    if type(rows) is not list or len(cast(list[object], rows)) != len(seeds):
        raise ValueError("NaP campaign rows drifted")
    checked_rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(cast(list[object], rows)):
        row_fields = {"seed", "initial_state_sha256", "result"}
        if type(raw_row) is not dict or set(cast(dict[str, object], raw_row)) != row_fields:
            raise ValueError("NaP campaign row fields drifted")
        row = cast(dict[str, object], raw_row)
        if type(row["seed"]) is not int or row["seed"] != seeds[index]:
            raise ValueError("NaP campaign row seed drifted")
        root = jr.key(seeds[index], impl="threefry2x32")
        _next_key, init_key = jr.split(root)
        expected_init = _state_sha256(_init_state(init_key, PROFILES[profile_id].hidden_width))
        if (
            type(row["initial_state_sha256"]) is not str
            or row["initial_state_sha256"] != expected_init
        ):
            raise ValueError("NaP campaign initialization identity drifted")
        result = _validate_result_payload(row["result"], seeds[index], profile_id)
        if result["dataset_sha256"] != identity["dataset_sha256"]:
            raise ValueError("NaP campaign row dataset drifted")
        expected_schedule = _schedule_sha(
            _schedule(checked_images, checked_labels, PROFILES[profile_id], seeds[index])
        )
        if result["schedule_sha256"] != expected_schedule:
            raise ValueError("NaP campaign row schedule identity drifted")
        if reexecute:
            replay = _result_payload(checked_images, checked_labels, seeds[index], profile_id)
            if _without_timing(result) != _without_timing(replay):
                raise ValueError("NaP campaign row differs from independent replay")
        checked_rows.append(row)
    if report["aggregate"] != _aggregate(checked_rows):
        raise ValueError("NaP campaign aggregate drifted")
    if report["policy"] != {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_results_retained": True,
    }:
        raise ValueError("NaP campaign policy drifted")
    unsigned = dict(report)
    claimed = unsigned.pop("sha256")
    if type(claimed) is not str or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("NaP campaign report digest drifted")


Reservation = tuple[int, str, str, int, int, int, str, int, int]


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("NaP campaign write made no progress")
        view = view[written:]


def _open_parent(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in absolute.parent.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("NaP campaign output has an unsafe path component")
            try:
                os.mkdir(component, 0o755, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_existing_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("NaP campaign output has an unsafe path component")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _reserve(path: Path) -> Reservation:
    if type(path) is not type(Path()) or path.absolute() != OUTPUT_PATH.absolute():
        raise ValueError("NaP campaign output must be the exact registered path")
    directory = _open_parent(path)
    marker = f".{path.name}.reservation"
    marker_fd = -1
    try:
        probe = os.open(
            ".", os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC, 0o400, dir_fd=directory
        )
        os.close(probe)
        try:
            os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("NaP campaign output already exists")
        marker_fd = os.open(
            marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400, dir_fd=directory
        )
        _write_all(marker_fd, b"asi-nap-campaign-reserved-v1\n")
        os.fsync(marker_fd)
        metadata = os.fstat(marker_fd)
        parent_metadata = os.fstat(directory)
        os.fsync(directory)
        return (
            directory,
            path.name,
            marker,
            marker_fd,
            metadata.st_dev,
            metadata.st_ino,
            os.fspath(path.absolute().parent),
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        )
    except BaseException:
        if marker_fd >= 0:
            try:
                metadata = os.fstat(marker_fd)
                visible = os.stat(marker, dir_fd=directory, follow_symlinks=False)
                if (visible.st_dev, visible.st_ino) == (metadata.st_dev, metadata.st_ino):
                    os.unlink(marker, dir_fd=directory)
                    os.fsync(directory)
            except FileNotFoundError:
                pass
            os.close(marker_fd)
        os.close(directory)
        raise


def _owned(reservation: Reservation) -> None:
    directory, _name, marker, marker_fd, device, inode, _parent, _parent_dev, _parent_ino = (
        reservation
    )
    held = os.fstat(marker_fd)
    visible = os.stat(marker, dir_fd=directory, follow_symlinks=False)
    if (
        not stat.S_ISREG(held.st_mode)
        or (held.st_dev, held.st_ino) != (device, inode)
        or (visible.st_dev, visible.st_ino) != (device, inode)
    ):
        raise RuntimeError("NaP campaign reservation identity changed")


def _assert_visible_parent(reservation: Reservation) -> None:
    directory, _name, _marker, _marker_fd, _device, _inode, parent, device, inode = (
        reservation
    )
    held = os.fstat(directory)
    visible_fd = _open_existing_directory(Path(parent))
    try:
        visible = os.fstat(visible_fd)
    finally:
        os.close(visible_fd)
    if (
        not stat.S_ISDIR(held.st_mode)
        or (held.st_dev, held.st_ino) != (device, inode)
        or (visible.st_dev, visible.st_ino) != (device, inode)
    ):
        raise RuntimeError("NaP registered output parent identity changed")


def _finish(reservation: Reservation, *, consumed: bool) -> None:
    directory, _name, marker, marker_fd, _device, _inode, _parent, _parent_dev, _parent_ino = (
        reservation
    )
    try:
        _owned(reservation)
        if consumed:
            os.ftruncate(marker_fd, 0)
            os.lseek(marker_fd, 0, os.SEEK_SET)
            _write_all(marker_fd, b"asi-nap-consumed-without-result-v1\n")
            os.fsync(marker_fd)
        else:
            os.unlink(marker, dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(marker_fd)
        os.close(directory)


def _link_tmpfile(file_fd: int, directory_fd: int, name: str) -> None:
    linkat = ctypes.CDLL(None, use_errno=True).linkat
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    if linkat(file_fd, b"", directory_fd, os.fsencode(name), 0x1000) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), name)
        raise OSError(number, os.strerror(number), name)


def _publish(
    reservation: Reservation, report: dict[str, object], images: np.ndarray, labels: np.ndarray
) -> None:
    directory, name, _marker, _marker_fd, _device, _inode, _parent, _parent_dev, _parent_ino = (
        reservation
    )
    validate_report(
        report, images, labels, seeds=CAMPAIGN_RESERVED_SEEDS, profile_id=PROFILE_ID,
        reexecute=True
    )
    encoded = _canonical(report) + b"\n"
    descriptor = os.open(".", os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC, 0o400, dir_fd=directory)
    published: tuple[int, int] | None = None
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 0:
            raise ValueError("NaP staged campaign inode drifted")
        _owned(reservation)
        _assert_visible_parent(reservation)
        published = metadata.st_dev, metadata.st_ino
        _link_tmpfile(descriptor, directory, name)
        os.fsync(directory)
        _assert_visible_parent(reservation)
        read_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        try:
            raw = os.read(read_fd, len(encoded) + 1)
        finally:
            os.close(read_fd)
        if raw != encoded:
            raise ValueError("NaP published campaign changed during strict reread")
        def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("NaP published campaign contains duplicate keys")
                result[key] = item
            return result

        def invalid_constant(value: str) -> object:
            raise ValueError(f"invalid JSON constant {value}")

        loaded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=exact_object,
            parse_constant=invalid_constant,
        )
        validate_report(
            loaded, images, labels, seeds=CAMPAIGN_RESERVED_SEEDS, profile_id=PROFILE_ID,
            reexecute=False
        )
    except BaseException:
        if published is not None:
            try:
                visible = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (visible.st_dev, visible.st_ino) == published:
                    os.unlink(name, dir_fd=directory)
                    os.fsync(directory)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(descriptor)


def run_and_publish(data_home: Path = default_openml_data_home()) -> dict[str, object]:
    if _REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True:
        raise PermissionError("NaP campaign execution is not authorized")
    reservation = _reserve(OUTPUT_PATH)
    dispatched = False
    published = False
    try:
        images, labels = _load_canonical_dataset(data_home)
        def note_dispatch() -> None:
            nonlocal dispatched
            dispatched = True

        report = _run(
            images,
            labels,
            seeds=CAMPAIGN_RESERVED_SEEDS,
            profile_id=PROFILE_ID,
            on_dispatch=note_dispatch,
        )
        _publish(reservation, report, images, labels)
        published = True
        return report
    finally:
        _finish(reservation, consumed=dispatched and not published)
