"""Prospectively frozen, permanently nonpromoting AdamO matched screen.

Execution is closed until a separate reviewed authorization change flips
``_EXECUTION_AUTHORIZED`` for the reserved matrix.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from alberta_framework.benchmarks.adamo_diagnostic import (
    _MATCHED_EXECUTION_CAPABILITY,
    ARMS,
    COMPARISON_ID,
    FROZEN_MATCHED_DEVELOPMENT_SEEDS,
    OFFICIAL_CODE,
    OFFICIAL_CODE_SEARCH_DATE,
    PAPER_URL,
    _run_matched_adamo_diagnostic,
    validate_adamo_diagnostic,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    _screening_dataset_provenance,
    _screening_runtime_environment,
    _screening_source_provenance,
    _validated_dataset_provenance,
    _validated_runtime_environment,
    _validated_source_provenance,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    default_openml_data_home,
    load_mnist_train,
)
from alberta_framework.core.adamo import ADAMO_PAPER_REVISION

SCHEMA: Final[str] = "asi.adamo-matched-development.report.v1"
PLAN_ID: Final[str] = "issue-1560-adamo-bounded-development-v1"
PROFILE: Final[str] = "bounded-development"
SEEDS: Final[tuple[int, ...]] = FROZEN_MATCHED_DEVELOPMENT_SEEDS
CONSUMED_QUALIFICATION_SEEDS: Final[tuple[int, ...]] = (15600, 15601, 15602, 15603)
QUARANTINED_PREPLAN_SEEDS: Final[tuple[int, ...]] = (25600, 25601, 25602, 25603)
CONTROL_ARM: Final[str] = "adamw_control"
CANDIDATE_ARMS: Final[tuple[str, ...]] = ("adamo_l1e3", "adam_iso_joint_l1e3")
T95_DF3: Final[float] = 3.1824463052837078
DATASET_NUMERIC_BYTES: Final[int] = 188_400_000
CANONICAL_X_SHA256: Final[str] = (
    "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
)
CANONICAL_Y_SHA256: Final[str] = (
    "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
)
_EXECUTION_AUTHORIZED: Final[bool] = False
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Final[Path] = _REPO_ROOT / "outputs/adamo_matched_development/report.v1.json"
_MAX_JSON_NODES: Final[int] = 100_000
_MAX_JSON_STRING_BYTES: Final[int] = 2 * 1024 * 1024
_MAX_REPORT_BYTES: Final[int] = 32 * 1024 * 1024


@dataclasses.dataclass(frozen=True, slots=True)
class _OutputReservation:
    directory_fd: int
    destination_name: str
    reservation_name: str


def frozen_plan() -> dict[str, object]:
    """Return the literal plan that must be merged before any execution."""
    return {
        "plan_id": PLAN_ID,
        "paper_revision": ADAMO_PAPER_REVISION,
        "paper_url": PAPER_URL,
        "official_code": OFFICIAL_CODE,
        "official_code_search_date": OFFICIAL_CODE_SEARCH_DATE,
        "official_parity_status": "blocked_no_author_maintained_code_located",
        "comparison_id": COMPARISON_ID,
        "profile": PROFILE,
        "arms": list(ARMS),
        "control_arm": CONTROL_ARM,
        "candidate_arms": list(CANDIDATE_ARMS),
        "seeds": list(SEEDS),
        "consumed_qualification_seeds": list(CONSUMED_QUALIFICATION_SEEDS),
        "quarantined_preplan_seeds": list(QUARANTINED_PREPLAN_SEEDS),
        "qualification_seed_note": (
            "15600-15603 were exercised by contract qualification and are excluded from "
            "the retained matched matrix; the exposed 25600-25603 preplan roster is also "
            "quarantined because 25600 was exercised by an earlier test fixture"
        ),
        "primary_metric": "mean_online_accuracy",
        "selection_arm": "adamo_l1e3",
        "causal_ablation_arm": "adam_iso_joint_l1e3",
        "hypothesis": (
            "the decoupled AdamO isometry step reduces Jacobian/Gram degradation and yields "
            "a positive paired mean-online-accuracy delta against AdamW"
        ),
        "failure_condition": (
            "the AdamO accuracy interval is not wholly positive, the inert reduction fails, "
            "or any frozen identity/resource invariant fails"
        ),
        "paired_direction": "higher_is_better",
        "confidence_method": "two_sided_student_t",
        "confidence_level": 0.95,
        "confidence_degrees_of_freedom": 3,
        "confidence_critical": T95_DF3,
        "multiple_comparison_policy": (
            "only adamo_l1e3 informs the development disposition; the joint-gradient arm is "
            "a descriptive causal ablation"
        ),
        "matched_axes": [
            "dataset",
            "initialization_root",
            "task_permutations",
            "example_schedule",
            "observations",
            "updates",
            "allowed_learner_information",
        ],
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "diagnostic_information": ["post_task_boundary_index", "fixed_input_row_0"],
        "dataset": {
            "source": {
                "provider": "openml",
                "name": "mnist_784",
                "version": 1,
                "row_start": 0,
                "row_stop_exclusive": 60000,
            },
            "numeric_bytes": DATASET_NUMERIC_BYTES,
            "arrays": {
                "x": {
                    "dtype": "<f4",
                    "shape": [60000, 784],
                    "sha256": CANONICAL_X_SHA256,
                },
                "y": {
                    "dtype": "<i4",
                    "shape": [60000],
                    "sha256": CANONICAL_Y_SHA256,
                },
            },
            "materialization": (
                "OpenML mnist_784 v1 rows 0:60000; float32 pixels scaled by "
                "(x / 255 - 0.5) / 0.5 and int32 labels"
            ),
        },
        "paper_protocol_differences": [
            "IPMNIST adaptation, not reproduction of a paper task",
            "784-300-150-10 ReLU MLP instead of the paper depth-4 width-512 MLP",
            "existing protocol initialization instead of paper orthogonal initialization",
            "eight tasks of 64 updates instead of a paper training horizon",
            "no convolutional, RL, transformer, GroupSort, Newton-Schulz, NTK, or rank protocol",
        ],
        "mechanism_off_reduction": "adamo_inert == adamw_control bit-exact",
        "resource_policy": (
            "observations, updates, data/environment steps, model queries, Jacobian rows, "
            "persistent numeric bytes, peak Gram workspace, logical compute, and telemetry "
            "timing are retained per arm and seed"
        ),
        "output_path": "outputs/adamo_matched_development/report.v1.json",
        "execution_authorized": _EXECUTION_AUTHORIZED,
        "execution_status": (
            "authorized_after_separate_review"
            if _EXECUTION_AUTHORIZED
            else "blocked_pending_independent_plan_audit"
        ),
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retention_required": True,
    }


def _digest(value: object, *, context: str, length: int = 64) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase hexadecimal digest")
    return value


def _exact_object(
    value: object, expected_keys: frozenset[str], *, context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    mapping = cast(dict[object, object], value)
    raw_keys = tuple(mapping)
    if (
        len(raw_keys) != len(expected_keys)
        or any(type(key) is not str for key in raw_keys)
        or frozenset(cast(tuple[str, ...], raw_keys)) != expected_keys
    ):
        raise ValueError(f"{context} keys do not match the frozen schema")
    return cast(dict[str, Any], mapping)


def _finite_float(value: object, *, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be an exact finite float")
    return value


def _bounded_json(value: object, *, context: str) -> object:
    """Copy exact JSON without invoking subclass hooks or unbounded traversal."""
    budget = [0, 0]

    def visit(item: object, *, depth: int, label: str) -> object:
        budget[0] += 1
        if budget[0] > _MAX_JSON_NODES or depth > 18:
            raise ValueError(f"{context} exceeds the JSON structure bound")
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError(f"{label} exceeds signed-int64")
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{label} must be finite")
            return item
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(f"{label} must be valid UTF-8") from error
            if len(encoded) > 16_384 or b"\0" in encoded:
                raise ValueError(f"{label} must be a bounded string")
            budget[1] += len(encoded)
            if budget[1] > _MAX_JSON_STRING_BYTES:
                raise ValueError(f"{context} exceeds the JSON string-byte bound")
            return item
        if type(item) is list:
            if list.__len__(item) > 4096:
                raise ValueError(f"{label} exceeds the list bound")
            return [
                visit(child, depth=depth + 1, label=f"{label}[{index}]")
                for index, child in enumerate(list.__iter__(item))
            ]
        if type(item) is dict:
            if dict.__len__(item) > 4096:
                raise ValueError(f"{label} exceeds the object bound")
            result: dict[str, object] = {}
            for key, child in dict.items(item):
                if type(key) is not str:
                    raise ValueError(f"{label} keys must be exact strings")
                result[key] = visit(child, depth=depth + 1, label=f"{label}.{key}")
            return result
        raise ValueError(f"{label} must contain only exact JSON values")

    return visit(value, depth=0, label=context)


def _current_source_provenance() -> dict[str, object]:
    return _screening_source_provenance(_REPO_ROOT)


def _current_runtime_environment() -> dict[str, object]:
    return _screening_runtime_environment()


def _validated_canonical_dataset_provenance(
    value: object, *, context: str
) -> dict[str, Any]:
    provenance = _validated_dataset_provenance(value, context=context)
    x_binding = cast(dict[str, object], provenance["x"])
    y_binding = cast(dict[str, object], provenance["y"])
    if (
        x_binding["sha256"] != CANONICAL_X_SHA256
        or y_binding["sha256"] != CANONICAL_Y_SHA256
    ):
        raise ValueError(
            f"{context} dataset does not match the frozen canonical OpenML materialization"
        )
    return provenance


def _validate_plan(value: object) -> dict[str, object]:
    expected = frozen_plan()
    plan = _exact_object(value, frozenset(expected), context="plan")
    normalized = _bounded_json(plan, context="plan")
    if normalized != expected:
        raise ValueError("report plan does not equal the literal frozen plan")
    return cast(dict[str, object], normalized)


def _arm_by_name(receipt: Mapping[str, object], name: str) -> dict[str, object]:
    arms = receipt["arms"]
    if type(arms) is not list:
        raise ValueError("receipt arms must be an exact list")
    for raw_arm in arms:
        if type(raw_arm) is dict and raw_arm.get("arm") == name:
            return cast(dict[str, object], raw_arm)
    raise ValueError(f"receipt is missing arm {name}")


def _mean_accuracy(receipt: Mapping[str, object], arm: str) -> float:
    values = _arm_by_name(receipt, arm)["per_task_accuracy"]
    if type(values) is not list or not values:
        raise ValueError("per_task_accuracy must be a nonempty exact list")
    converted = tuple(_finite_float(value, context="per_task_accuracy") for value in values)
    return math.fsum(converted) / len(converted)


def _final_diagnostic(receipt: Mapping[str, object], arm: str, key: str) -> float:
    diagnostics = _arm_by_name(receipt, arm)["post_task_diagnostics"]
    if type(diagnostics) is not list or not diagnostics or type(diagnostics[-1]) is not dict:
        raise ValueError("post_task_diagnostics must be a nonempty exact object list")
    return _finite_float(
        cast(dict[str, object], diagnostics[-1])[key], context=f"final {key}"
    )


def _resource_deltas(receipt: Mapping[str, object], candidate: str) -> dict[str, int]:
    control_resources = _arm_by_name(receipt, CONTROL_ARM)["resources"]
    candidate_resources = _arm_by_name(receipt, candidate)["resources"]
    if type(control_resources) is not dict or type(candidate_resources) is not dict:
        raise ValueError("arm resources must be exact objects")
    keys = (
        "observations",
        "updates",
        "data_steps",
        "environment_steps",
        "model_queries",
        "jacobian_reverse_rows",
        "persistent_numeric_bytes",
        "peak_gram_working_bytes",
        "logical_compute_units",
    )
    deltas: dict[str, int] = {}
    for key in keys:
        control = control_resources.get(key)
        value = candidate_resources.get(key)
        if type(control) is not int or type(value) is not int:
            raise ValueError(f"resource {key} must be an exact integer")
        deltas[key] = value - control
    return deltas


def _paired_comparison(
    receipts: Mapping[int, Mapping[str, object]], candidate: str
) -> dict[str, object]:
    accuracy_deltas = tuple(
        _mean_accuracy(receipts[seed], candidate)
        - _mean_accuracy(receipts[seed], CONTROL_ARM)
        for seed in SEEDS
    )
    mean = math.fsum(accuracy_deltas) / len(accuracy_deltas)
    centered = math.fsum((value - mean) ** 2 for value in accuracy_deltas)
    standard_error = math.sqrt(centered / (len(accuracy_deltas) - 1)) / math.sqrt(
        len(accuracy_deltas)
    )
    lower = mean - T95_DF3 * standard_error
    upper = mean + T95_DF3 * standard_error
    outcome = "supported" if lower > 0.0 else "rejected" if upper <= 0.0 else "inconclusive"
    rms_deltas = [
        _final_diagnostic(receipts[seed], candidate, "jacobian_rms_distance_from_one")
        - _final_diagnostic(receipts[seed], CONTROL_ARM, "jacobian_rms_distance_from_one")
        for seed in SEEDS
    ]
    gram_deltas = [
        _final_diagnostic(receipts[seed], candidate, "weight_gram_penalty")
        - _final_diagnostic(receipts[seed], CONTROL_ARM, "weight_gram_penalty")
        for seed in SEEDS
    ]
    resource_deltas = _resource_deltas(receipts[SEEDS[0]], candidate)
    if any(_resource_deltas(receipts[seed], candidate) != resource_deltas for seed in SEEDS[1:]):
        raise ValueError("candidate resource deltas must be seed-invariant")
    return {
        "accuracy_deltas": list(accuracy_deltas),
        "mean_accuracy_delta": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "outcome": outcome,
        "final_jacobian_rms_distance_deltas": rms_deltas,
        "final_weight_gram_penalty_deltas": gram_deltas,
        "resource_deltas_vs_control": resource_deltas,
    }


def build_report(
    receipts: Sequence[dict[str, object]],
    *,
    dataset_provenance: Mapping[str, object],
    source_provenance: Mapping[str, object],
    runtime_environment: Mapping[str, object],
) -> dict[str, object]:
    """Build and strictly validate the complete seed-by-arm campaign report."""
    if (type(receipts) is not list and type(receipts) is not tuple) or len(receipts) != len(
        SEEDS
    ):
        raise ValueError("receipts must contain the complete frozen seed schedule")
    by_seed: dict[int, dict[str, object]] = {}
    ordered: list[dict[str, object]] = []
    for index, raw_receipt in enumerate(receipts):
        receipt = validate_adamo_diagnostic(
            raw_receipt, seed_schedule=SEEDS, require_current_identity=True
        )
        seed = receipt["seed"]
        if type(seed) is not int or seed != SEEDS[index] or seed in by_seed:
            raise ValueError("receipts must use deterministic frozen seed ordering")
        if receipt["profile"] != PROFILE:
            raise ValueError("receipt profile does not match the frozen plan")
        by_seed[seed] = receipt
        ordered.append(receipt)
    normalized_dataset = _validated_canonical_dataset_provenance(
        dataset_provenance, context="AdamO report"
    )
    normalized_source = _validated_source_provenance(
        source_provenance, context="AdamO report"
    )
    normalized_runtime = _validated_runtime_environment(
        runtime_environment, context="AdamO report"
    )
    if normalized_source != _current_source_provenance():
        raise ValueError("source provenance does not match the current source")
    if normalized_runtime != _current_runtime_environment():
        raise ValueError("runtime environment does not match the current runtime")
    source_receipts = [item["source"] for item in ordered]
    runtime_receipts = [item["runtime"] for item in ordered]
    if any(value != source_receipts[0] for value in source_receipts[1:]):
        raise ValueError("all receipts must bind one source identity")
    if any(value != runtime_receipts[0] for value in runtime_receipts[1:]):
        raise ValueError("all receipts must bind one runtime identity")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "plan": frozen_plan(),
        "dataset_provenance": normalized_dataset,
        "source_provenance": normalized_source,
        "runtime_environment": normalized_runtime,
        "runs": ordered,
        "paired_comparisons": {
            candidate: _paired_comparison(by_seed, candidate) for candidate in CANDIDATE_ARMS
        },
        "development_disposition": "inconclusive",
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "outcome_retained": True,
            "timing_is_telemetry_only": True,
        },
    }
    comparisons = cast(dict[str, object], report["paired_comparisons"])
    primary_comparison = cast(dict[str, object], comparisons["adamo_l1e3"])
    report["development_disposition"] = primary_comparison["outcome"]
    return validate_report(report, require_current_execution_identity=True)


def validate_report(
    payload: object, *, require_current_execution_identity: bool = False
) -> dict[str, object]:
    """Validate a retained report offline, optionally requiring this execution identity."""
    if type(require_current_execution_identity) is not bool:
        raise ValueError("require_current_execution_identity must be an exact bool")
    if type(payload) is not dict:
        raise TypeError("report must be an exact dict")
    normalized_payload = _bounded_json(payload, context="report")
    report = _exact_object(
        normalized_payload,
        frozenset(
            {
                "schema",
                "plan",
                "dataset_provenance",
                "source_provenance",
                "runtime_environment",
                "runs",
                "paired_comparisons",
                "development_disposition",
                "policy",
            }
        ),
        context="report",
    )
    if type(report["schema"]) is not str or report["schema"] != SCHEMA:
        raise ValueError("report schema does not match the frozen protocol")
    _validate_plan(report["plan"])
    dataset_provenance = _validated_canonical_dataset_provenance(
        report["dataset_provenance"], context="AdamO report"
    )
    source = _validated_source_provenance(
        report["source_provenance"], context="AdamO report"
    )
    runtime = _validated_runtime_environment(
        report["runtime_environment"], context="AdamO report"
    )
    if require_current_execution_identity and source != _current_source_provenance():
        raise ValueError("report source provenance does not match current source")
    if require_current_execution_identity and runtime != _current_runtime_environment():
        raise ValueError("report runtime environment does not match the current runtime")
    policy = _exact_object(
        report["policy"],
        frozenset(
            {
                "development_only",
                "scientific_promotion_allowed",
                "outcome_retained",
                "timing_is_telemetry_only",
            }
        ),
        context="policy",
    )
    if any(type(value) is not bool for value in policy.values()) or policy != {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retained": True,
        "timing_is_telemetry_only": True,
    }:
        raise ValueError("report policy must remain permanently nonpromoting")
    runs = report["runs"]
    if type(runs) is not list or len(runs) != len(SEEDS):
        raise ValueError("runs must contain the complete frozen seed schedule")
    by_seed: dict[int, dict[str, object]] = {}
    run_source_identity: object | None = None
    combined_dataset_identity: object | None = None
    for index, raw_run in enumerate(runs):
        run = validate_adamo_diagnostic(
            raw_run,
            seed_schedule=SEEDS,
            require_current_identity=require_current_execution_identity,
        )
        seed = run["seed"]
        if type(seed) is not int or seed != SEEDS[index] or seed in by_seed:
            raise ValueError("runs must use deterministic frozen seed ordering")
        if run["profile"] != PROFILE:
            raise ValueError("run profile does not match the frozen plan")
        run_dataset = cast(dict[str, object], run["dataset"])
        x_binding = cast(dict[str, object], dataset_provenance["x"])
        y_binding = cast(dict[str, object], dataset_provenance["y"])
        if (
            run_dataset["x_sha256"] != x_binding["sha256"]
            or run_dataset["y_sha256"] != y_binding["sha256"]
            or run_dataset["rows"] != cast(list[int], x_binding["shape"])[0]
        ):
            raise ValueError("run dataset identity does not match the aggregate")
        if combined_dataset_identity is None:
            combined_dataset_identity = run_dataset["sha256"]
        elif run_dataset["sha256"] != combined_dataset_identity:
            raise ValueError("runs do not bind one combined dataset identity")
        if run_source_identity is None:
            run_source_identity = run["source"]
        elif run["source"] != run_source_identity:
            raise ValueError("runs do not bind one diagnostic source identity")
        run_runtime = cast(dict[str, object], run["runtime"])
        python_binding = cast(dict[str, object], runtime["python"])
        packages = cast(dict[str, object], runtime["packages"])
        jax_binding = cast(dict[str, object], runtime["jax"])
        expected_run_runtime = {
            "python": python_binding["version"],
            "jax": packages["jax"],
            "numpy": packages["numpy"],
            "backend": jax_binding["backend"],
        }
        if run_runtime != expected_run_runtime:
            raise ValueError("run runtime identity does not match the aggregate")
        by_seed[seed] = run
    paired = _exact_object(
        report["paired_comparisons"], frozenset(CANDIDATE_ARMS), context="paired comparisons"
    )
    normalized_paired = _bounded_json(paired, context="paired comparisons")
    expected_paired = {
        candidate: _paired_comparison(by_seed, candidate) for candidate in CANDIDATE_ARMS
    }
    if normalized_paired != expected_paired:
        raise ValueError("paired arithmetic does not match the retained runs")
    disposition = report["development_disposition"]
    if (
        type(disposition) is not str
        or disposition != expected_paired["adamo_l1e3"]["outcome"]
    ):
        raise ValueError("development disposition must equal the primary paired outcome")
    return cast(dict[str, object], report)


def _open_pinned_parent(path: Path) -> int:
    """Open or create the final directory without following its leaf name."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(path.parent, flags)
    except FileNotFoundError:
        grandparent_fd = os.open(path.parent.parent, flags)
        try:
            try:
                os.mkdir(path.parent.name, mode=0o755, dir_fd=grandparent_fd)
                os.fsync(grandparent_fd)
            except FileExistsError:
                pass
            return os.open(path.parent.name, flags, dir_fd=grandparent_fd)
        finally:
            os.close(grandparent_fd)


def _reserve_output(path: Path) -> _OutputReservation:
    """Acquire an O_EXCL sibling before execution and pin the destination directory."""
    if type(path) is not type(Path()):
        raise ValueError("output path must be an exact pathlib.Path")
    if path.absolute() != OUTPUT_PATH.absolute():
        raise ValueError(f"output path must be the reserved NEW path {OUTPUT_PATH}")
    if not path.name or path.name in {".", ".."}:
        raise ValueError("output path must have one exact destination name")
    directory_fd = _open_pinned_parent(path)
    reservation_name = f".{path.name}.reservation"
    reservation_fd = -1
    try:
        reservation_fd = os.open(
            reservation_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
            dir_fd=directory_fd,
        )
        reservation_bytes = f"reserved:{path.name}\n".encode("ascii")
        written = os.write(reservation_fd, reservation_bytes)
        if written != len(reservation_bytes):
            raise OSError("short write while reserving immutable output")
        os.fsync(reservation_fd)
        os.close(reservation_fd)
        reservation_fd = -1
        os.fsync(directory_fd)
        try:
            os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite immutable output: {path}")
        return _OutputReservation(directory_fd, path.name, reservation_name)
    except BaseException:
        if reservation_fd >= 0:
            os.close(reservation_fd)
        try:
            os.unlink(reservation_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
        raise


def _release_reservation(reservation: _OutputReservation) -> None:
    try:
        os.unlink(reservation.reservation_name, dir_fd=reservation.directory_fd)
        os.fsync(reservation.directory_fd)
    except FileNotFoundError:
        pass
    finally:
        os.close(reservation.directory_fd)


def _strict_reread(reservation: _OutputReservation, expected: bytes) -> object:
    descriptor = os.open(
        reservation.destination_name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=reservation.directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= _MAX_REPORT_BYTES:
            raise ValueError("published report is not one bounded regular file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("published report ended before its signed size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("published report grew during strict reread")
        actual = b"".join(chunks)
    finally:
        os.close(descriptor)
    if actual != expected:
        raise ValueError("published report bytes differ from the validated generation")
    try:
        parsed = json.loads(actual)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("published report is not strict JSON") from error
    return _bounded_json(parsed, context="published report")


def _publish_reserved(reservation: _OutputReservation, report: dict[str, object]) -> None:
    encoded = (
        json.dumps(report, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("validated report exceeds the immutable publication bound")
    temporary_name = f".{reservation.destination_name}.{secrets.token_hex(16)}.tmp"
    temporary_fd = -1
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
            dir_fd=reservation.directory_fd,
        )
        view = memoryview(encoded)
        while view:
            count = os.write(temporary_fd, view)
            if count <= 0:
                raise OSError("immutable report write made no progress")
            view = view[count:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        os.link(
            temporary_name,
            reservation.destination_name,
            src_dir_fd=reservation.directory_fd,
            dst_dir_fd=reservation.directory_fd,
            follow_symlinks=False,
        )
        os.fsync(reservation.directory_fd)
        reread = _strict_reread(reservation, encoded)
        if reread != report:
            raise ValueError("published report semantic reread differs from validation")
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=reservation.directory_fd)
            os.fsync(reservation.directory_fd)
        except FileNotFoundError:
            pass


def publish_report(path: Path, report: object) -> Path:
    """Reserve, validate, and no-replace publish one immutable generation."""
    if not _EXECUTION_AUTHORIZED:
        raise RuntimeError("AdamO matched-development publication is not authorized")
    reservation = _reserve_output(path)
    try:
        validated = validate_report(report, require_current_execution_identity=True)
        _publish_reserved(reservation, validated)
    finally:
        _release_reservation(reservation)
    return path


def run_campaign(
    data_home: Path, output_path: Path = OUTPUT_PATH
) -> dict[str, object]:
    """Execute exactly once after the separately reviewed authorization change."""
    if not _EXECUTION_AUTHORIZED:
        raise RuntimeError("AdamO matched-development execution is not authorized")
    if type(data_home) is not type(Path()) or type(output_path) is not type(Path()):
        raise ValueError("data home and output paths must be exact pathlib.Path values")
    if output_path.absolute() != OUTPUT_PATH.absolute():
        raise ValueError(f"output path must be the reserved NEW path {OUTPUT_PATH}")
    reservation = _reserve_output(output_path)
    try:
        source_before = _current_source_provenance()
        runtime_before = _current_runtime_environment()
        inputs, labels = load_mnist_train(data_home)
        dataset_before = _screening_dataset_provenance(inputs, labels)
        _validated_canonical_dataset_provenance(
            dataset_before, context="AdamO execution"
        )
        receipts = [
            _run_matched_adamo_diagnostic(
                inputs,
                labels,
                profile=PROFILE,
                seed=seed,
                capability=_MATCHED_EXECUTION_CAPABILITY,
            )
            for seed in SEEDS
        ]
        if source_before != _current_source_provenance():
            raise RuntimeError("source identity changed during matched execution")
        if runtime_before != _current_runtime_environment():
            raise RuntimeError("runtime identity changed during matched execution")
        if dataset_before != _screening_dataset_provenance(inputs, labels):
            raise RuntimeError("dataset identity changed during matched execution")
        report = build_report(
            receipts,
            dataset_provenance=dataset_before,
            source_provenance=source_before,
            runtime_environment=runtime_before,
        )
        _publish_reserved(reservation, report)
        return report
    finally:
        _release_reservation(reservation)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--data-home", type=Path, default=default_openml_data_home())
    args = parser.parse_args(argv)
    if args.catalog:
        print(json.dumps(frozen_plan(), sort_keys=True))
        return 0
    run_campaign(args.data_home)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
